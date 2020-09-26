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


def build_td3_critic(action_dim, hidden_units):
    return hk.transform(
        lambda x: ContinuousQFunction(
            num_critics=2,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


def build_td3_actor(action_dim, hidden_units):
    return hk.transform(
        lambda x: DeterministicPolicy(
            action_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


class TD3(ContinuousOffPolicyAlgorithm):
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
        units_actor=(256, 256),
        units_critic=(256, 256),
        std=0.1,
        std_target=0.2,
        clip_noise=0.5,
        update_interval_policy=2,
    ):
        super(TD3, self).__init__(
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
        self.critic = build_td3_critic(
            action_dim=action_space.shape[0],
            hidden_units=units_actor,
        )
        opt_init_critic, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(
            next(self.rng), np.zeros((1, state_space.shape[0] + action_space.shape[0]), np.float32)
        )
        self.opt_state_critic = opt_init_critic(self.params_critic)

        # Actor.
        self.actor = build_td3_actor(
            action_dim=action_space.shape[0],
            hidden_units=units_actor,
        )
        opt_init_actor, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.params_actor_target = self.actor.init(
            next(self.rng), np.zeros((1, *state_space.shape), np.float32)
        )
        self.opt_state_actor = opt_init_actor(self.params_actor)

        # Other parameters.
        self.std = std
        self.std_target = std_target
        self.clip_noise = clip_noise
        self.update_interval_policy = update_interval_policy

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
            rng=next(self.rng),
        )

        # Update actor.
        if self.learning_steps % self.update_interval_policy == 0:
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
        rng: jnp.ndarray,
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
            rng=rng,
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
        rng: jnp.ndarray,
    ) -> jnp.DeviceArray:
        next_action = self.actor.apply(params_actor_target, None, next_state)
        noise = jax.random.normal(rng, next_action.shape) * self.std_target
        next_action = jnp.clip(next_action + jnp.clip(noise, -self.clip_noise, self.clip_noise), -1.0, 1.0)
        next_q1, next_q2 = self.critic.apply(params_critic_target, None, jnp.concatenate([next_state, next_action], axis=1))
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * jnp.minimum(next_q1, next_q2))
        curr_q1, curr_q2 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        return jnp.square(target_q - curr_q1).mean() + jnp.square(target_q - curr_q2).mean()

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
        q1 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))[0]
        return -q1.mean()

    def __str__(self):
        return "td3"
