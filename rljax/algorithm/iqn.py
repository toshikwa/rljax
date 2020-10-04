import os
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import QLearning
from rljax.network import DiscreteImplicitQuantileFunction
from rljax.util import calculate_quantile_loss, get_quantile_at_action, load_params, save_params


class IQN(QLearning):
    name = "IQN"

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
        update_interval_target=8000,
        eps=0.01,
        eps_eval=0.001,
        eps_decay_steps=250000,
        lr=5e-5,
        units=(512,),
        num_quantiles=64,
        num_quantiles_eval=32,
        num_cosines=64,
        loss_type="l2",
        dueling_net=False,
        double_q=False,
    ):
        assert loss_type in ["l2", "huber"]
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
            eps_decay_steps=eps_decay_steps,
        )

        def quantile_fn(s, cum_p):
            return DiscreteImplicitQuantileFunction(
                action_space=action_space,
                num_critics=1,
                hidden_units=units,
                dueling_net=dueling_net,
            )(s, cum_p)

        # Quantile network.
        fake_cum_p = np.empty((1, num_quantiles), dtype=np.float32)
        self.quantile_net = hk.without_apply_rng(hk.transform(quantile_fn))
        self.params = self.params_target = self.quantile_net.init(next(self.rng), self.fake_state, fake_cum_p)
        opt_init, self.opt = optix.adam(lr, eps=0.01 / batch_size)
        self.opt_state = opt_init(self.params)

        # Other parameters.
        self.num_quantiles = num_quantiles
        self.num_quantiles_eval = num_quantiles_eval
        self.num_cosines = num_cosines
        self.loss_type = loss_type
        self.double_q = double_q

    def forward(self, state):
        return self._forward(self.params, next(self.rng), state)

    @partial(jax.jit, static_argnums=0)
    def _forward(
        self,
        params: hk.Params,
        key: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        cum_p = jax.random.uniform(key, (1, self.num_quantiles_eval))
        q_s = self.quantile_net.apply(params, state, cum_p).mean(axis=1)
        return jnp.argmax(q_s, axis=1)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        self.opt_state, self.params, loss, abs_td = self._update(
            opt_state=self.opt_state,
            params=self.params,
            params_target=self.params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            key1=next(self.rng),
            key2=next(self.rng),
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update target network.
        if self.agent_step % self.update_interval_target == 0:
            self.params_target = self._update_target(self.params_target, self.params)

        if writer and self.learning_step % 1000 == 0:
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
        key1: np.ndarray,
        key2: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss, abs_td), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            params_target=params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            key1=key1,
            key2=key2,
        )
        update, opt_state = self.opt(grad, opt_state)
        params = optix.apply_updates(params, update)
        return opt_state, params, loss, abs_td

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
        key1: np.ndarray,
        key2: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Sample cumulative probabilities.
        cum_p1 = jax.random.uniform(key1, (state.shape[0], self.num_quantiles))
        cum_p2 = jax.random.uniform(key2, (state.shape[0], self.num_quantiles))

        if self.double_q:
            # Calculate greedy actions with online network. (NOTE: We reuse key1 here for the simple implementation.)
            next_action = self._forward(params, key1, next_state)[..., None]
            # Then calculate max quantile values with target network.
            next_quantile = get_quantile_at_action(self.quantile_net.apply(params_target, next_state, cum_p2), next_action)
        else:
            # Calculate greedy actions and max quantile values with target network.
            next_quantile = jnp.max(self.quantile_net.apply(params_target, next_state, cum_p2), axis=2, keepdims=True)

        # Calculate target quantile values and reshape to (batch_size, 1, N).
        target_quantile = jnp.expand_dims(reward, 2) + (1.0 - jnp.expand_dims(done, 2)) * self.discount * next_quantile
        target_quantile = jax.lax.stop_gradient(target_quantile).reshape(-1, 1, self.num_quantiles)

        # Calculate current quantile values, whose shape is (batch_size, N, 1).
        curr_quantile = get_quantile_at_action(self.quantile_net.apply(params, state, cum_p1), action)
        td = target_quantile - curr_quantile
        loss = calculate_quantile_loss(td, cum_p1, weight, self.loss_type)
        abs_td = jnp.abs(td).sum(axis=1).mean(axis=1, keepdims=True)
        return loss, jax.lax.stop_gradient(abs_td)

    def save_params(self, save_dir):
        super(IQN, self).save_params(save_dir)
        save_params(self.params, os.path.join(save_dir, "params.npz"))

    def load_params(self, save_dir):
        self.params = self.params_target = load_params(os.path.join(save_dir, "params.npz"))
