from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.sac import SAC
from rljax.network import CategoricalPolicy, DiscreteQFunction
from rljax.util import get_q_at_action


class SAC_Discrete(SAC):
    name = "SAC-Discrete"

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
        batch_size=64,
        start_steps=10000,
        update_interval=4,
        update_interval_target=8000,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(512,),
        units_critic=(512,),
        target_entropy_ratio=0.98,
        dueling_net=False,
    ):
        if fn_critic is None:

            def fn_critic(s):
                return DiscreteQFunction(
                    action_space=action_space,
                    num_critics=2,
                    hidden_units=units_critic,
                    dueling_net=dueling_net,
                )(s)

        if fn_actor is None:

            def fn_actor(s):
                return CategoricalPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                )(s)

        # Entropy coefficient.
        self.target_entropy = -np.log(1.0 / action_space.n) * target_entropy_ratio

        # Other parameters.
        if not hasattr(self, "random_update_critic"):
            # SAC_Discrete._loss_critic() doesn't need a random key.
            self.random_update_critic = False
        if not hasattr(self, "random_update_actor"):
            # SAC_Discrete._loss_actor() doesn't need a random key.
            self.random_update_actor = False

        super(SAC_Discrete, self).__init__(
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
            update_interval_target=update_interval_target,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
        )

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        pi, _ = self.actor.apply(params_actor, state)
        return jnp.argmax(pi, axis=1)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        key: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        pi, _ = self.actor.apply(params_actor, state)
        return jax.random.categorical(key, pi)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_pi: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        next_q_s_list = self.critic.apply(params_critic_target, next_state)
        next_q = (next_pi * (jnp.asarray(next_q_s_list).min(axis=0) - alpha * next_log_pi)).sum(axis=1, keepdims=True)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_pi, next_log_pi = self.actor.apply(params_actor, next_state)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_pi, next_log_pi)
        curr_q_list = [get_q_at_action(curr_q_s, action) for curr_q_s in self.critic.apply(params_critic, state)]
        loss = 0.0
        for curr_q in curr_q_list:
            loss += (jnp.square(target - curr_q) * weight).mean()
        abs_td = jax.lax.stop_gradient(jnp.abs(target - curr_q[0]))
        return loss, abs_td

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        curr_q_s = jax.lax.stop_gradient(jnp.asarray(self.critic.apply(params_critic, state)).min(axis=0))
        pi, log_pi = self.actor.apply(params_actor, state)
        mean_q = (pi * curr_q_s).sum(axis=1).mean()
        mean_log_pi = (pi * log_pi).sum(axis=1).mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)
