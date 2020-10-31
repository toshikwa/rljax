from functools import partial
from typing import List, Tuple

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
        num_critics=2,
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
                    num_critics=num_critics,
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

        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = False
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = False

        super(SAC_Discrete, self).__init__(
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
        pi_s, _ = self.actor.apply(params_actor, state)
        return jnp.argmax(pi_s, axis=1)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        pi_s, _ = self.actor.apply(params_actor, state)
        return jax.random.categorical(key, pi_s)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.actor.apply(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> List[jnp.ndarray]:
        return [get_q_at_action(q_s, action) for q_s in self.critic.apply(params_critic, state)]

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        pi_s: np.ndarray,
    ) -> jnp.ndarray:
        q_s = jax.lax.stop_gradient(jnp.asarray(self.critic.apply(params_critic, state)).min(axis=0))
        return (pi_s * q_s).sum(axis=1, keepdims=True)

    @partial(jax.jit, static_argnums=0)
    def _calculate_log_pi(
        self,
        pi_s: np.ndarray,
        log_pi_s: np.ndarray,
    ) -> jnp.ndarray:
        return (pi_s * log_pi_s).sum(axis=1, keepdims=True)
