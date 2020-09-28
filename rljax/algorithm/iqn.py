from functools import partial
from typing import Any, Tuple

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import QLearning
from rljax.network.critic import DiscreteImplicitQuantileFunction, DQNBody
from rljax.util import calculate_quantile_huber_loss, get_quantile_at_action


def build_iqn(state_space, action_space, num_quantiles, num_cosines, feature_dim, hidden_units, dueling_net):
    def _func(state, tau):
        if len(state_space.shape) == 3:
            state_feature = DQNBody()(state)
        elif len(state_space.shape) == 1:
            state_feature = nn.relu(hk.Linear(feature_dim)(state))
        else:
            NotImplementedError

        return DiscreteImplicitQuantileFunction(
            action_dim=action_space.n,
            num_critics=1,
            num_quantiles=num_quantiles,
            num_cosines=num_cosines,
            feature_dim=state_feature.shape[1],
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
            dueling_net=dueling_net,
        )(state_feature, tau)

    return hk.without_apply_rng(hk.transform(_func))


class IQN(QLearning):
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
        num_quantiles=64,
        num_cosines=64,
        feature_dim=32,  # Ignored when the state is an image.
        dueling_net=True,
        double_q=True,
    ):
        assert update_interval_target % update_interval == 0
        super(IQN, self).__init__(
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

        # IQN.
        fake_tau = np.empty((1, num_quantiles), dtype=np.float32)
        self.quantile_net = build_iqn(state_space, action_space, num_quantiles, num_cosines, feature_dim, units, dueling_net)
        opt_init, self.opt = optix.adam(lr)
        self.params = self.params_target = self.quantile_net.init(next(self.rng), self.fake_state, fake_tau)
        self.opt_state = opt_init(self.params)

        # Other parameters.
        self.double_q = double_q
        self.num_quantiles = num_quantiles
        self.num_cosines = num_cosines

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        tau = jax.random.uniform(rng, (1, self.num_quantiles))
        q = self.quantile_net.apply(params, state, tau).mean(axis=1)
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
            rng=next(self.rng),
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
        rng: np.ndarray,
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
            rng=rng,
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
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Sample fractions from uniform distributions.
        tau = jax.random.uniform(rng, (state.shape[0], self.num_quantiles))
        if self.double_q:
            # Calculate greedy actions with online network.
            next_action = jnp.argmax(self.quantile_net.apply(params, next_state, tau).mean(axis=1), axis=1)[..., None]
            # Then calculate max quantile values with target network.
            next_quantile = get_quantile_at_action(self.quantile_net.apply(params_target, next_state, tau), next_action)
        else:
            # Calculate greedy actions and max quantile values with target network.
            next_quantile = jnp.max(self.quantile_net.apply(params_target, next_state, tau), axis=2, keepdims=True)

        # Calculate target quantile values and reshape to (batch_size, 1, N).
        target_quantile = jnp.expand_dims(reward, 2) + (1.0 - jnp.expand_dims(done, 2)) * self.discount * next_quantile
        target_quantile = jax.lax.stop_gradient(target_quantile).reshape(-1, 1, self.num_quantiles)
        # Calculate current quantile values, whose shape is (batch_size, N, 1).
        curr_quantile = get_quantile_at_action(self.quantile_net.apply(params, state, tau), action)
        td = target_quantile - curr_quantile
        loss = calculate_quantile_huber_loss(td, tau, weight, 1.0)
        error = jnp.abs(td).sum(axis=1).mean(axis=1, keepdims=True)
        return loss, jax.lax.stop_gradient(error)

    def __str__(self):
        return "IQN" if not self.use_per else "IQN+PER"
