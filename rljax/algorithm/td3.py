from functools import partial
from typing import Any, Tuple

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network.actor import DeterministicPolicy
from rljax.network.critic import ContinuousQFunction
from rljax.utils import add_noise


def build_td3_critic(hidden_units):
    def _func(state, action):
        return ContinuousQFunction(
            num_critics=2,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(state, action)

    return hk.without_apply_rng(hk.transform(_func))


def build_td3_actor(action_space, hidden_units):
    def _func(state):
        return DeterministicPolicy(
            action_dim=action_space.shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(state)

    return hk.without_apply_rng(hk.transform(_func))


class TD3(OffPolicyActorCritic):
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
        batch_size=128,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        lr_actor=1e-3,
        lr_critic=1e-3,
        units_actor=(400, 300),
        units_critic=(400, 300),
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
    ):
        super(TD3, self).__init__(
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

        # Critic.
        self.critic = build_td3_critic(units_critic)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), self.fake_state, self.fake_action)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = build_td3_actor(action_space, units_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), self.fake_state)
        self.opt_state_actor = opt_init(self.params_actor)

        # Other parameters.
        self.std = std
        self.std_target = std_target
        self.clip_noise = clip_noise
        self.update_interval_policy = update_interval_policy

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        return self.actor.apply(params_actor, state)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, state)
        return add_noise(action, rng, self.std, -1.0, 1.0)

    def update(self, writer):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic and target.
        self.opt_state_critic, self.params_critic, loss_critic, error = self._update_critic(
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
            rng=next(self.rng),
        )
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(error)

        if self.learning_step % self.update_interval_policy == 0:
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
        rng: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_critic, error), grad_critic = jax.value_and_grad(self._loss_critic, has_aux=True)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor_target=params_actor_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            rng=rng,
        )
        update, opt_state_critic = self.opt_critic(grad_critic, opt_state_critic)
        params_critic = optix.apply_updates(params_critic, update)
        return opt_state_critic, params_critic, loss_critic, error

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
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate next actions and add clipped noises.
        next_action = self.actor.apply(params_actor_target, next_state)
        noise = jax.random.normal(rng, next_action.shape) * self.std_target
        next_action = jnp.clip(next_action + jnp.clip(noise, -self.clip_noise, self.clip_noise), -1.0, 1.0)
        # Calculate target q values (clipped double q) with target critic.
        next_q1, next_q2 = self.critic.apply(params_critic_target, next_state, next_action)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * jnp.minimum(next_q1, next_q2))
        # Calculate current q values with online critic.
        curr_q1, curr_q2 = self.critic.apply(params_critic, state, action)
        error = jnp.abs(target_q - curr_q1)
        loss = (jnp.square(error) * weight).mean() + (jnp.square(target_q - curr_q2) * weight).mean()
        return loss, jax.lax.stop_gradient(error)

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
        q1 = self.critic.apply(params_critic, state, action)[0]
        return -q1.mean()

    def __str__(self):
        return "TD3"
