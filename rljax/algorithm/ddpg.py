import os
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network import ContinuousQFunction, DeterministicPolicy
from rljax.util import add_noise, load_params, save_params


class DDPG(OffPolicyActorCritic):
    name = "DDPG"

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
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        lr_actor=1e-3,
        lr_critic=1e-3,
        units_actor=(256, 256),
        units_critic=(256, 256),
        std=0.1,
        update_interval_policy=2,
    ):
        super(DDPG, self).__init__(
            num_steps=num_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
        )

        def critic_fn(s, a):
            return ContinuousQFunction(
                num_critics=1,
                hidden_units=units_critic,
            )(s, a)

        def actor_fn(s):
            return DeterministicPolicy(
                action_space=action_space,
                hidden_units=units_actor,
            )(s)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), self.fake_state, self.fake_action)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), self.fake_state)
        opt_init, self.opt_actor = optix.adam(lr_actor)
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
        key: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        return add_noise(action, key, self.std, -1.0, 1.0)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic and target.
        self.opt_state_critic, self.params_critic, loss_critic, abs_td = self._update_critic(
            opt_state_critic=self.opt_state_critic,
            params_critic=self.params_critic,
            params_actor_target=self.params_actor_target,
            params_critic_target=self.params_critic_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        if writer and self.learning_step % self.update_interval_policy == 0:
            # Update actor and target.
            self.opt_state_actor, self.params_actor, loss_actor = self._update_actor(
                opt_state_actor=self.opt_state_actor,
                params_actor=self.params_actor,
                params_critic=self.params_critic,
                state=state,
            )
            self.params_actor_target = self._update_target(self.params_actor_target, self.params_actor)

            if self.learning_step % 1000 == 0:
                writer.add_scalar("loss/critic", loss_critic, self.learning_step)
                writer.add_scalar("loss/actor", loss_actor, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic: Any,
        params_critic: hk.Params,
        params_actor_target: hk.Params,
        params_critic_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_critic, abs_td), grad_critic = jax.value_and_grad(self._loss_critic, has_aux=True)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor_target=params_actor_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )
        update, opt_state_critic = self.opt_critic(grad_critic, opt_state_critic)
        params_critic = optix.apply_updates(params_critic, update)
        return opt_state_critic, params_critic, loss_critic, abs_td

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate next actions.
        next_action = self.actor.apply(params_actor_target, next_state)
        # Calculate target q values with target critic.
        next_q = self.critic.apply(params_critic_target, next_state, next_action)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        # Calculate current q values with online critic.
        curr_q = self.critic.apply(params_critic, state, action)
        abs_td = jnp.abs(target_q - curr_q)
        loss = (jnp.square(abs_td) * weight).mean()
        return loss, jax.lax.stop_gradient(abs_td)

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor: Any,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray]:
        loss_actor, grad_actor = jax.value_and_grad(self._loss_actor)(
            params_actor,
            params_critic=params_critic,
            state=state,
        )
        update, opt_state_actor = self.opt_actor(grad_actor, opt_state_actor)
        params_actor = optix.apply_updates(params_actor, update)
        return opt_state_actor, params_actor, loss_actor

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        q = self.critic.apply(params_critic, state, action)
        return -q.mean()

    def save_params(self, save_dir):
        super(DDPG, self).save_params(save_dir)
        save_params(self.params_critic, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_critic = self.params_critic_target = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = self.params_actor_target = load_params(os.path.join(save_dir, "params_actor.npz"))
