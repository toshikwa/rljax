from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.qrdqn import QRDQN
from rljax.network import DiscreteImplicitQuantileFunction
from rljax.util import fake_state


class IQN(QRDQN):
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
        loss_type="huber",
        dueling_net=False,
        double_q=False,
        setup_net=True,
        fn=None,
        lr=5e-5,
        units=(512,),
        num_quantiles=64,
        num_quantiles_eval=32,
        num_cosines=64,
    ):
        if fn is None:

            def fn(s, cum_p):
                return DiscreteImplicitQuantileFunction(
                    num_cosines=num_cosines,
                    action_space=action_space,
                    hidden_units=units,
                    dueling_net=dueling_net,
                )(s, cum_p)

        if not hasattr(self, "fake_args"):
            self.fake_args = (fake_state(state_space), np.empty((1, num_quantiles), dtype=np.float32))
        if not hasattr(self, "use_key_forward"):
            self.use_key_forward = True
        if not hasattr(self, "num_keys_loss"):
            self.num_keys_loss = 3

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
            loss_type=loss_type,
            dueling_net=dueling_net,
            double_q=double_q,
            setup_net=setup_net,
            fn=fn,
            lr=lr,
            num_quantiles=num_quantiles,
        )
        self.num_quantiles_eval = num_quantiles_eval

    @partial(jax.jit, static_argnums=0)
    def _forward(
        self,
        params: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        cum_p = jax.random.uniform(key, (state.shape[0], self.num_quantiles_eval))
        return jnp.argmax(self.net.apply(params, state, cum_p).mean(axis=1), axis=1)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params: hk.Params,
        params_target: hk.Params,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        cum_p_prime: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        if self.double_q:
            next_action = self._forward(params, next_state, key=key)[..., None]
        else:
            next_action = self._forward(params_target, next_state, key=key)[..., None]
        next_quantile = self._calculate_value(params_target, next_state, next_action, cum_p=cum_p_prime)
        target = reward[:, None] + (1.0 - done[:, None]) * self.discount * next_quantile
        return jax.lax.stop_gradient(target).reshape(-1, 1, self.num_quantiles)

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
        key_list: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cum_p = jax.random.uniform(key_list[0], (state.shape[0], self.num_quantiles))
        cum_p_prime = jax.random.uniform(key_list[1], (state.shape[0], self.num_quantiles))
        quantile = self._calculate_value(params, state, action, cum_p=cum_p)
        target = self._calculate_target(params, params_target, reward, done, next_state, cum_p_prime, key_list[2])
        return self._calculate_loss_and_abs_td(quantile, target, cum_p, weight)
