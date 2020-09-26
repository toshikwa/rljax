from functools import partial
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
from flax import nn, optim
from rljax.common.base_class import ContinuousOnPolicyAlgorithm
from rljax.common.utils import clip_gradient, update_network
from rljax.ppo.network import build_ppo_actor, build_ppo_critic


def calculate_gae(
    critic: nn.Model,
    gamma: float,
    lambd: float,
    state: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    next_state: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    value = jax.lax.stop_gradient(critic(state))
    next_value = jax.lax.stop_gradient(critic(next_state))

    # Calculate TD errors.
    delta = reward + gamma * next_value * (1.0 - done) - value

    # Calculate gae recursively from behind.
    gae = [delta[-1]]
    for t in jnp.arange(reward.shape[0] - 2, -1, -1):
        gae.insert(0, delta[t] + gamma * lambd * (1 - done[t]) * gae[0])
    gae = jnp.array(gae)

    return gae + value, (gae - gae.mean()) / (gae.std() + 1e-8)


def critic_grad_fn(
    critic: nn.Model,
    max_grad_norm: float,
    state: jnp.ndarray,
    target: jnp.ndarray,
) -> nn.Model:
    def critic_loss_fn(critic):
        return jnp.square(critic(state) - target).mean()

    grad_critic = jax.grad(critic_loss_fn)(critic)
    return clip_gradient(grad_critic, max_grad_norm)


def actor_grad_fn(
    actor: nn.Model,
    critic: nn.Model,
    clip_eps: float,
    max_grad_norm: float,
    log_pi_old: jnp.ndarray,
    gae: jnp.ndarray,
    state: jnp.ndarray,
    action: jnp.ndarray,
) -> nn.Model:
    def actor_loss_fn(actor):
        log_pi = actor(state, action=action)
        ratio = jnp.exp(log_pi - log_pi_old)
        loss_actor1 = -ratio * gae
        loss_actor2 = -jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
        loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
        return loss_actor

    grad_actor = jax.grad(actor_loss_fn)(actor)
    return clip_gradient(grad_actor, max_grad_norm)


class PPO(ContinuousOnPolicyAlgorithm):
    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma=0.995,
        buffer_size=1000,
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(64, 64),
        units_critic=(64, 64),
        epoch_ppo=32,
        clip_eps=0.2,
        lambd=0.97,
        max_grad_norm=10.0,
    ):
        super(PPO, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            buffer_size=buffer_size,
        )

        # Actor.
        actor = build_ppo_actor(
            state_dim=state_space.shape[0],
            action_dim=action_space.shape[0],
            rng_init=next(self.rng),
            hidden_units=units_actor,
        )
        self.optim_actor = jax.device_put(optim.Adam(learning_rate=lr_actor).create(actor))

        # Critic.
        critic = build_ppo_critic(
            state_dim=state_space.shape[0],
            rng_init=next(self.rng),
            hidden_units=units_critic,
        )
        self.optim_critic = jax.device_put(optim.Adam(learning_rate=lr_critic).create(critic))

        # Compile functions.
        self.calculate_gae = jax.jit(partial(calculate_gae, gamma=gamma, lambd=lambd))
        self.critic_grad_fn = jax.jit(partial(critic_grad_fn, max_grad_norm=max_grad_norm))
        self.actor_grad_fn = jax.jit(partial(actor_grad_fn, clip_eps=clip_eps, max_grad_norm=max_grad_norm))

        # Other parameters.
        self.epoch_ppo = epoch_ppo

    def select_action(self, state):
        state = jax.device_put(state[None, ...])
        action = self.actor(state, deterministic=True)
        return np.array(action[0])

    def explore(self, state):
        state = jax.device_put(state[None, ...])
        action, log_pi = self.actor(state, key=next(self.rng), deterministic=False)
        return np.array(action[0]), np.array(log_pi[0])

    def update(self):
        state, action, reward, done, log_pi_old, next_state = self.buffer.get()

        # Calculate gamma-return and gae.
        target, gae = self.calculate_gae(
            critic=self.critic,
            state=state,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        for _ in range(self.epoch_ppo):
            self.learning_steps += 1
            # Update critic.
            grad_critic = self.critic_grad_fn(
                critic=self.critic,
                state=state,
                target=target,
            )
            self.optim_critic = update_network(self.optim_critic, grad_critic)

            # Update actor.
            grad_actor = self.actor_grad_fn(
                actor=self.actor,
                critic=self.critic,
                log_pi_old=log_pi_old,
                gae=gae,
                state=state,
                action=action,
            )
            self.optim_actor = update_network(self.optim_actor, grad_actor)

    @property
    def actor(self):
        return self.optim_actor.target

    @property
    def critic(self):
        return self.optim_critic.target

    def __str__(self):
        return "ppo"
