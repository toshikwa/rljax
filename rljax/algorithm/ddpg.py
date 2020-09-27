from functools import partial

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


def build_ddpg_critic(action_dim, hidden_units):
    return hk.transform(
        lambda x: ContinuousQFunction(
            num_critics=1,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


def build_ddpg_actor(action_dim, hidden_units):
    return hk.transform(
        lambda x: DeterministicPolicy(
            action_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


class DDPG(OffPolicyActorCritic):
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
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(400, 300),
        units_critic=(400, 300),
        std=0.1,
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

        # Critic.
        fake_input = np.zeros((1, state_space.shape[0] + action_space.shape[0]), np.float32)
        self.critic = build_ddpg_critic(action_space.shape[0], units_critic)
        opt_init_critic, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), fake_input)
        self.opt_state_critic = opt_init_critic(self.params_critic)

        # Actor.
        fake_input = np.zeros((1, *state_space.shape), np.float32)
        self.actor = build_ddpg_actor(action_space.shape[0], units_actor)
        opt_init_actor, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), fake_input)
        self.opt_state_actor = opt_init_actor(self.params_actor)

        # Other parameters.
        self.std = std

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        return self.actor.apply(params_actor, None, state)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        action = self.actor.apply(params_actor, None, state)
        return add_noise(action, rng, self.std, -1.0, 1.0)

    def update(self, writer):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
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
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(error)

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor = self._update_actor(
            opt_state_actor=self.opt_state_actor,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            state=state,
        )

        # Update target networks.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)
        self.params_actor_target = self._update_target(self.params_actor_target, self.params_actor)

        if self.learning_step % 1000 == 0:
            writer.add_scalar('loss/critic', loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic,
        params_critic: hk.Params,
        params_actor_target: hk.Params,
        params_critic_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ):
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
    ) -> jnp.ndarray:
        next_action = self.actor.apply(params_actor_target, None, next_state)
        next_q = self.critic.apply(params_critic_target, None, jnp.concatenate([next_state, next_action], axis=1))
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        error = jnp.abs(target_q - curr_q)
        loss = (jnp.square(error) * weight).mean()
        return loss, jax.lax.stop_gradient(error)

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ):
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
        action = self.actor.apply(params_actor, None, state)
        q = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        return -q.mean()

    def __str__(self):
        return "DDPG"
