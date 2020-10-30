from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.ddpg import DDPG
from rljax.network import ContinuousQFunction, DeterministicPolicy
from rljax.util import add_noise


class TD3(DDPG):
    name = "TD3"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=2,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=1e-3,
        lr_critic=1e-3,
        units_actor=(256, 256),
        units_critic=(256, 256),
        d2rl=False,
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
    ):
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = True

        super(TD3, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            std=std,
            update_interval_policy=update_interval_policy,
        )
        self.std_target = std_target
        self.clip_noise = clip_noise

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate next actions and add clipped noises.
        next_action = self.actor.apply(params_actor_target, next_state)
        next_action = add_noise(next_action, key, self.std_target, -1.0, 1.0, -self.clip_noise, self.clip_noise)
        # Calculate target q values (clipped double q) with target critic.
        next_q_list = self.critic.apply(params_critic_target, next_state, next_action)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * jnp.asarray(next_q_list).min(axis=0))
        # Calculate current q values with online critic.
        q_list = self.critic.apply(params_critic, state, action)
        loss = 0.0
        for q in q_list:
            loss += (jnp.square(target_q - q) * weight).mean()
        abs_td = jax.lax.stop_gradient(jnp.abs(target_q - q_list[0]))
        return loss, abs_td

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        q = self.critic.apply(params_critic, state, action)[0]
        return -q.mean(), None
