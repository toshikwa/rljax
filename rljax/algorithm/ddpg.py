from functools import partial

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import ContinuousOffPolicyAlgorithm
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


class DDPG(ContinuousOffPolicyAlgorithm):
    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        batch_size=256,
        start_steps=10000,
        tau=5e-3,
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(400, 300),
        units_critic=(400, 300),
        std=0.1,
    ):
        super(DDPG, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=False,
            batch_size=batch_size,
            start_steps=start_steps,
            tau=tau,
        )

        # Critic.
        fake_input = np.zeros((1, state_space.shape[0] + action_space.shape[0]), np.float32)
        self.critic = build_ddpg_critic(action_space.shape[0], units_actor)
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

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(self, params_actor, state):
        return self.actor.apply(params_actor, None, state)

    @partial(jax.jit, static_argnums=0)
    def _explore(self, params_actor, rng, state):
        action = self.actor.apply(params_actor, None, state)
        return add_noise(action, rng, self.std, -1.0, 1.0)

    def update(self):
        self.learning_steps += 1
        _, (state, action, reward, done, next_state) = self.buffer.sample(self.batch_size)

        # Update critic.
        self.opt_state_critic, self.params_critic = self._update_critic(
            opt_state_critic=self.opt_state_critic,
            params_critic=self.params_critic,
            params_actor_target=self.params_actor_target,
            params_critic_target=self.params_critic_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        # Update actor.
        self.opt_state_actor, self.params_actor = self._update_actor(
            opt_state_actor=self.opt_state_actor,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            state=state,
        )

        # Update target networks.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)
        self.params_actor_target = self._update_target(self.params_actor_target, self.params_actor)

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
    ):
        grad_critic = jax.grad(self._loss_critic)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor_target=params_actor_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )
        update, opt_state_critic = self.opt_critic(grad_critic, opt_state_critic)
        params_critic = optix.apply_updates(params_critic, update)
        return opt_state_critic, params_critic

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
    ) -> jnp.DeviceArray:
        next_action = self.actor.apply(params_actor_target, None, next_state)
        next_q = self.critic.apply(params_critic_target, None, jnp.concatenate([next_state, next_action], axis=1))
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        return jnp.square(target_q - curr_q).mean()

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ):
        grad_actor = jax.grad(self._loss_actor)(
            params_actor,
            params_critic=params_critic,
            state=state,
        )
        update, opt_state_actor = self.opt_actor(grad_actor, opt_state_actor)
        params_actor = optix.apply_updates(params_actor, update)
        return opt_state_actor, params_actor

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
    ) -> jnp.DeviceArray:
        action = self.actor.apply(params_actor, None, state)
        q = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        return -q.mean()

    def __str__(self):
        return "ddpg"
