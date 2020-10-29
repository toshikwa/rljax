from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.dqn import DQN
from rljax.network import DiscreteImplicitQuantileFunction
from rljax.util import get_quantile_at_action, quantile_loss


class IQN(DQN):
    name = "IQN"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
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
        fn=None,
        lr=5e-5,
        units=(512,),
        num_quantiles=64,
        num_quantiles_eval=32,
        num_cosines=64,
        loss_type="huber",
        dueling_net=False,
        double_q=False,
    ):
        assert loss_type in ["l2", "huber"]
        if fn is None:

            def fn(s, cum_p):
                return DiscreteImplicitQuantileFunction(
                    action_space=action_space,
                    num_critics=1,
                    hidden_units=units,
                    dueling_net=dueling_net,
                )(s, cum_p)

        fake_cum_p = np.empty((1, num_quantiles), dtype=np.float32)
        fake_state = state_space.sample()[None, ...]
        if len(state_space.shape) == 1:
            fake_state = fake_state.astype(np.float32)
        self.fake_args = (fake_state, fake_cum_p)

        if not hasattr(self, "random_update"):
            # IQN._loss() doesn't need a random key.
            self.random_update = True

        super(IQN, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
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
            fn=fn,
            lr=lr,
        )

        # Other parameters.
        self.num_quantiles = num_quantiles
        self.num_quantiles_eval = num_quantiles_eval
        self.num_cosines = num_cosines

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
        q_s = self.net.apply(params, state, cum_p).mean(axis=1)
        return jnp.argmax(q_s, axis=1)

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
            next_quantile = get_quantile_at_action(self.net.apply(params_target, next_state, cum_p2), next_action)
        else:
            # Calculate greedy actions and max quantile values with target network.
            next_quantile = jnp.max(self.net.apply(params_target, next_state, cum_p2), axis=2, keepdims=True)

        # Calculate target quantile values and reshape to (batch_size, 1, N).
        target_quantile = jnp.expand_dims(reward, 2) + (1.0 - jnp.expand_dims(done, 2)) * self.discount * next_quantile
        target_quantile = jax.lax.stop_gradient(target_quantile).reshape(-1, 1, self.num_quantiles)

        # Calculate current quantile values, whose shape is (batch_size, N, 1).
        curr_quantile = get_quantile_at_action(self.net.apply(params, state, cum_p1), action)
        td = target_quantile - curr_quantile
        loss = quantile_loss(td, cum_p1, weight, self.loss_type)
        abs_td = jnp.abs(td).sum(axis=1).mean(axis=1, keepdims=True)
        return loss, jax.lax.stop_gradient(abs_td)
