from functools import partial
from typing import Any, Tuple

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import QLearning
from rljax.network.critic import DiscreteQuantileFunction, DQNBody
from rljax.util import calculate_quantile_huber_loss, get_quantile_at_action


def build_qrdqn(state_space, action_space, num_quantiles, hidden_units, dueling_net):
    def _func(state):
        if len(state_space.shape) == 3:
            state = DQNBody()(state)
        return DiscreteQuantileFunction(
            action_dim=action_space.n,
            num_critics=1,
            num_quantiles=num_quantiles,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
            dueling_net=dueling_net,
        )(state)

    return hk.without_apply_rng(hk.transform(_func))


class QRDQN(QLearning):
    def __init__(
        self,
        num_steps,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=32,
        start_steps=50000,
        update_interval=4,
        update_interval_target=10000,
        eps=0.01,
        eps_eval=0.001,
        lr=2.5e-4,
        units=(512,),
        num_quantiles=200,
        dueling_net=True,
        double_q=True,
    ):
        assert update_interval_target % update_interval == 0
        super(QRDQN, self).__init__(
            num_steps=num_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            batch_size=batch_size,
            use_per=use_per,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
            eps=eps,
            eps_eval=eps_eval,
        )

        # QR-DQN.
        self.quantile_net = build_qrdqn(state_space, action_space, num_quantiles, units, dueling_net)
        opt_init, self.opt = optix.adam(lr)
        self.params = self.params_target = self.quantile_net.init(next(self.rng), self.fake_state)
        self.opt_state = opt_init(self.params)

        # Fixed fractions.
        tau = jnp.arange(0, num_quantiles + 1, dtype=jnp.float32) / num_quantiles
        self.tau_hat = jnp.expand_dims((tau[1:] + tau[:-1]) / 2.0, 0)

        # Other parameters.
        self.double_q = double_q
        self.num_quantiles = num_quantiles

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        q = self.quantile_net.apply(params, state).mean(axis=1)
        return jnp.argmax(q, axis=1)

    def update(self, writer):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        self.opt_state, self.params, loss, error = self._update(
            opt_state=self.opt_state,
            params=self.params,
            params_target=self.params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(error)

        # Update target network.
        if self.env_step % self.update_interval_target == 0:
            self.params_target = self._update_target(self.params_target, self.params)

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/quantile", loss, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update(
        self,
        opt_state: Any,
        params: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss, error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            params_target=params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )
        update, opt_state = self.opt(grad, opt_state)
        params = optix.apply_updates(params, update)
        return opt_state, params, loss, error

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.double_q:
            # calculate greedy actions with online network.
            next_action = jnp.argmax(self.quantile_net.apply(params, next_state).mean(axis=1), axis=1)[..., None]
            # Then calculate max quantile values with target network.
            next_quantile = get_quantile_at_action(self.quantile_net.apply(params_target, next_state), next_action)
        else:
            # calculate greedy actions and max quantile values with target network.
            next_quantile = jnp.max(self.quantile_net.apply(params_target, next_state), axis=2, keepdims=True)

        # calculate target quantile values and reshape to (batch_size, 1, N).
        target_quantile = jnp.expand_dims(reward, 2) + (1.0 - jnp.expand_dims(done, 2)) * self.discount * next_quantile
        target_quantile = jax.lax.stop_gradient(target_quantile).reshape(-1, 1, self.num_quantiles)
        # calculate current quantile values, whose shape is (batch_size, N, 1).
        curr_quantile = get_quantile_at_action(self.quantile_net.apply(params, state), action)
        loss, error = calculate_quantile_huber_loss(target_quantile - curr_quantile, self.tau_hat, weight, 1.0)
        return loss, jax.lax.stop_gradient(error)

    def __str__(self):
        return "QR-DQN" if not self.use_per else "QR-DQN+PER"
