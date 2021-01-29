from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rljax.algorithm.base_class import OffPolicyActorCritic
from rljax.network import ContinuousQFunction, DeterministicPolicy
from rljax.util import add_noise, optimize


class DDPG(OffPolicyActorCritic):
    name = "DDPG"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        num_critics=1,
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
        update_interval_policy=2,
    ):
        super(DDPG, self).__init__(
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
        )
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(s, a):
                return ContinuousQFunction(
                    num_critics=num_critics,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(s, a)

        if fn_actor is None:

            def fn_actor(s):
                return DeterministicPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    d2rl=d2rl,
                )(s)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(fn_critic))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_critic = optax.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), *self.fake_args_actor)
        opt_init, self.opt_actor = optax.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Other parameters.
        self.std = std
        self.update_interval_policy = update_interval_policy

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        return self.actor.apply(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        return add_noise(action, key, self.std, -1.0, 1.0)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic and target.
        self.opt_state_critic, self.params_critic, loss_critic, abs_td = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_actor_target=self.params_actor_target,
            params_critic_target=self.params_critic_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **self.kwargs_critic,
        )
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor and target.
        if self.learning_step % self.update_interval_policy == 0:
            self.opt_state_actor, self.params_actor, loss_actor, _ = optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
                params_critic=self.params_critic,
                state=state,
                **self.kwargs_actor,
            )
            self.params_actor_target = self._update_target(self.params_actor_target, self.params_actor)

            if writer and self.learning_step % 1000 == 0:
                writer.add_scalar("loss/critic", loss_critic, self.learning_step)
                writer.add_scalar("loss/actor", loss_actor, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _sample_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        return self.actor.apply(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
    ) -> jnp.ndarray:
        next_q = self._calculate_value(params_critic_target, next_state, next_action)
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

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
        *args,
        **kwargs,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action = self._sample_action(params_actor_target, next_state, *args, **kwargs)
        target = self._calculate_target(params_critic_target, reward, done, next_state, next_action)
        q_list = self._calculate_value_list(params_critic, state, action)
        return self._calculate_loss_critic_and_abs_td(q_list, target, weight)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        mean_q = self.critic.apply(params_critic, state, action)[0].mean()
        return -mean_q, None
