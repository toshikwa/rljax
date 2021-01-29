import os
from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rljax.network import ContinuousQFunction
from rljax.util import load_params, save_params


class DisCorMixIn:
    """
    MixIn for DisCor-based algorithms.
    """

    def __init__(
        self,
        num_critics=2,
        fn_error=None,
        lr_error=3e-4,
        units_error=(256, 256, 256),
        d2rl=False,
        init_error=10.0,
    ):
        if fn_error is None:

            def fn_error(s, a):
                return ContinuousQFunction(
                    num_critics=num_critics,
                    hidden_units=units_error,
                    d2rl=d2rl,
                )(s, a)

        # Error model.
        self.error = hk.without_apply_rng(hk.transform(fn_error))
        self.params_error = self.params_error_target = self.error.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_error = optax.adam(lr_error)
        self.opt_state_error = opt_init(self.params_error)
        # Running mean of error.
        self.rm_error_list = [jnp.array(init_error, dtype=jnp.float32) for _ in range(num_critics)]

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        q_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight_list: np.ndarray,
    ) -> jnp.ndarray:
        loss_critic = 0.0
        abs_td_list = []
        for q, weight in zip(q_list, weight_list):
            abs_td = jnp.abs(target - q_list[0])
            loss_critic += (jnp.square(abs_td) * weight).mean()
            abs_td_list.append(jax.lax.stop_gradient(abs_td))
        return loss_critic, abs_td

    @partial(jax.jit, static_argnums=0)
    def _calculate_error_list(
        self,
        params_actor: hk.Params,
        params_error: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action = self._explore(params_actor, state, key)
        return self.error.apply(params_error, state, action)

    @partial(jax.jit, static_argnums=0)
    def _calculate_weight_list(
        self,
        params_actor: hk.Params,
        params_error_target: hk.Params,
        rm_error_list: List[jnp.ndarray],
        done: np.ndarray,
        next_state: np.ndarray,
        key: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        next_error_list = self._calculate_error_list(params_actor, params_error_target, next_state, key)
        weight_list = []
        for next_error, rm_error in zip(next_error_list, rm_error_list):
            x = -(1.0 - done) * self.gamma * next_error / rm_error
            weight_list.append(jax.lax.stop_gradient(jax.nn.softmax(x, axis=0) * x.shape[0]))
        return weight_list

    @partial(jax.jit, static_argnums=0)
    def _loss_error(
        self,
        params_error: hk.Params,
        params_error_target: hk.Params,
        params_actor: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        abs_td_list: List[jnp.ndarray],
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        error_list = self.error.apply(params_error, state, action)
        next_error_list = self._calculate_error_list(params_actor, params_error_target, next_state, key)
        loss_error = 0.0
        mean_error_list = []
        for error, next_error, abs_td in zip(error_list, next_error_list, abs_td_list):
            target = jax.lax.stop_gradient(abs_td + (1.0 - done) * self.gamma * next_error)
            loss_error += jnp.square(error - target).mean()
            mean_error_list.append(jax.lax.stop_gradient(error.mean()))
        return loss_error, mean_error_list

    def save_params(self, save_dir):
        save_params(self.params_error, os.path.join(save_dir, "params_error.npz"))

    def load_params(self, save_dir):
        self.params_error = self.params_error_target = load_params(os.path.join(save_dir, "params_error.npz"))
