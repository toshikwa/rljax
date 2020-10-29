from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQuantileFunction, StateDependentGaussianPolicy
from rljax.util import quantile_loss


class TQC(SAC):
    name = "TQC"

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
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(512, 512, 512),
        d2rl=False,
        num_critics=5,
        num_quantiles=25,
        num_quantiles_to_drop=0,
    ):
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(s, a):
                return ContinuousQuantileFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    num_quantiles=num_quantiles,
                    d2rl=d2rl,
                )(s, a)

        if fn_actor is None:

            def fn_actor(s):
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    d2rl=d2rl,
                )(s)

        super(TQC, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
        )

        # Other parameters.
        cum_p = jnp.arange(0, num_quantiles + 1, dtype=jnp.float32) / num_quantiles
        self.cum_p_prime = jnp.expand_dims((cum_p[1:] + cum_p[:-1]) / 2.0, 0)
        self.num_quantiles = num_quantiles
        self.num_quantiles_target = num_quantiles * num_critics - num_quantiles_to_drop

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        next_quantile = jnp.concatenate(self.critic.apply(params_critic_target, next_state, next_action), axis=1)
        next_quantile = jnp.sort(next_quantile)[:, : self.num_quantiles_target] - alpha * next_log_pi
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_quantile)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action, next_log_pi = self._sample_action(params_actor, key, next_state)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_action, next_log_pi)
        curr_quantile_list = self.critic.apply(params_critic, state, action)
        loss = 0.0
        for curr_quantile in curr_quantile_list:
            loss += quantile_loss(target[:, None, :] - curr_quantile[:, :, None], self.cum_p_prime, weight, "huber")
        abs_td = jnp.abs(target[:, None, :] - curr_quantile_list[0][:, :, None]).sum(axis=1).mean(axis=1, keepdims=True)
        return loss, abs_td

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        key: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        # Sample actions.
        action, log_pi = self._sample_action(params_actor, key, state)
        # Calculate soft q values with online critic.
        mean_q = jnp.concatenate(self.critic.apply(params_critic, state, action), axis=1).mean()
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)
