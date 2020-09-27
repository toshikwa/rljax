from functools import partial
from typing import Any

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import QLearning
from rljax.network.critic import DiscreteQuantileFunction
from rljax.utils import calculate_quantile_huber_loss, get_quantile_at_action


def build_qrdqn(action_dim, num_quantiles, hidden_units, dueling_net):
    return hk.transform(
        lambda x: DiscreteQuantileFunction(
            action_dim=action_dim,
            num_quantiles=num_quantiles,
            num_critics=1,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
            dueling_net=dueling_net,
        )(x)
    )


class QRDQN(QLearning):
    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=256,
        start_steps=1000,
        update_interval=1,
        update_interval_target=1000,
        eps=0.01,
        eps_eval=0.001,
        lr=1e-4,
        units=(512,),
        num_quantiles=200,
        dueling_net=False,
        double_q=True,
    ):
        assert update_interval_target % update_interval == 0
        super(QRDQN, self).__init__(
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
        fake_input = np.zeros((1, state_space.shape[0]), np.float32)
        self.quantile_net = build_qrdqn(action_space.n, num_quantiles, units, dueling_net)
        opt_init, self.opt = optix.adam(lr)
        self.params = self.params_target = self.quantile_net.init(next(self.rng), fake_input)
        self.opt_state = opt_init(self.params)

        # Fixed fractions.
        tau = jnp.arange(0, num_quantiles + 1, dtype=jnp.float32) / num_quantiles
        self.tau_hat = jnp.expand_dims((tau[1:] + tau[:-1]) / 2.0, 0)

        # Other parameters.
        self.double_q = double_q
        self.num_quantiles = num_quantiles

    @partial(jax.jit, static_argnums=0)
    def _select_action(self, params, state):
        q = self.quantile_net.apply(params, None, state).mean(axis=1)
        return jnp.argmax(q, axis=1)

    def update(self):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        self.opt_state, self.params, error = self._update(
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
    ):
        grad, error = jax.grad(self._loss, has_aux=True)(
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
        return opt_state, params, error

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
    ) -> jnp.ndarray:
        if self.double_q:
            # calculate greedy actions with online network.
            next_action = jnp.argmax(self.quantile_net.apply(params, None, next_state).mean(axis=1), axis=1)[..., None]
            # Then calculate max quantile values with target network.
            next_quantile = get_quantile_at_action(self.quantile_net.apply(params_target, None, next_state), next_action)
        else:
            # calculate greedy actions and max quantile values with target network.
            next_quantile = jnp.max(self.quantile_net.apply(params_target, None, next_state), axis=2, keepdims=True)

        target_quantile = jnp.expand_dims(reward, 2) + (1.0 - jnp.expand_dims(done, 2)) * self.discount * next_quantile
        target_quantile = jax.lax.stop_gradient(target_quantile).reshape(-1, 1, self.num_quantiles)
        curr_quantile = get_quantile_at_action(self.quantile_net.apply(params, None, state), action)

        loss, error = calculate_quantile_huber_loss(target_quantile - curr_quantile, self.tau_hat, weight, 1.0)
        return loss, jax.lax.stop_gradient(error)

    def __str__(self):
        return "QR-DQN" if not self.use_per else "QR-DQN+PER"
