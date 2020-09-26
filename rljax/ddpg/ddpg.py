from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from flax import nn, optim
from rljax.common.base_class import ContinuousOffPolicyAlgorithm
from rljax.common.utils import add_noise, soft_update, update_network
from rljax.ddpg.network import build_ddpg_actor, build_ddpg_critic


def critic_grad_fn(
    critic: nn.Model,
    actor_target: nn.Model,
    critic_target: nn.Model,
    discount: float,
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    next_state: jnp.ndarray,
) -> nn.Model:
    next_action = actor_target(next_state)
    next_q = critic_target(next_state, next_action)
    target_q = jax.lax.stop_gradient(reward + (1.0 - done) * discount * next_q)

    def critic_loss_fn(critic):
        curr_q = critic(state, action)
        loss_critic = jnp.square(target_q - curr_q).mean()
        return loss_critic

    grad_critic = jax.grad(critic_loss_fn)(critic)
    return grad_critic


def actor_grad_fn(
    actor: nn.Model,
    critic: nn.Model,
    state: jnp.ndarray,
) -> nn.Model:
    def actor_loss_fn(actor):
        loss_actor = -critic(state, actor(state)).mean()
        return loss_actor

    grad_actor = jax.grad(actor_loss_fn)(actor)
    return grad_actor


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

        # Actor.
        rng_actor = next(self.rng)
        actor = build_ddpg_actor(
            state_dim=state_space.shape[0],
            action_dim=action_space.shape[0],
            rng_init=rng_actor,
            hidden_units=units_actor,
        )
        self.optim_actor = jax.device_put(optim.Adam(learning_rate=lr_actor).create(actor))

        # Critic.
        rng_critic = next(self.rng)
        critic = build_ddpg_critic(
            state_dim=state_space.shape[0],
            action_dim=action_space.shape[0],
            rng_init=rng_critic,
            hidden_units=units_critic,
        )
        self.optim_critic = jax.device_put(optim.Adam(learning_rate=lr_critic).create(critic))

        # Target networks.
        self.actor_target = jax.device_put(
            build_ddpg_actor(
                state_dim=state_space.shape[0],
                action_dim=action_space.shape[0],
                rng_init=rng_actor,
                hidden_units=units_critic,
            )
        )
        self.critic_target = jax.device_put(
            build_ddpg_critic(
                state_dim=state_space.shape[0],
                action_dim=action_space.shape[0],
                rng_init=rng_critic,
                hidden_units=units_critic,
            )
        )

        # Compile functions.
        self.critic_grad_fn = jax.jit(partial(critic_grad_fn, discount=self.discount))
        self.actor_grad_fn = jax.jit(actor_grad_fn)

        # Other parameters.
        self.std = std

    def select_action(self, state):
        state = jax.device_put(state[None, ...])
        action = self.actor(state)
        return np.array(action[0])

    def explore(self, state):
        state = jax.device_put(state[None, ...])
        action = self.actor(state)
        action = add_noise(action, next(self.rng), self.std, -1.0, 1.0)
        return np.array(action[0])

    def update(self):
        self.learning_steps += 1
        _, (state, action, reward, done, next_state) = self.buffer.sample(self.batch_size)

        # Update critic.
        grad_critic = self.critic_grad_fn(
            critic=self.critic,
            actor_target=self.actor_target,
            critic_target=self.critic_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )
        self.optim_critic = update_network(self.optim_critic, grad_critic)

        # Update actor.
        grad_actor = self.actor_grad_fn(
            actor=self.actor,
            critic=self.critic,
            state=state,
        )
        self.optim_actor = update_network(self.optim_actor, grad_actor)

        # Update target networks.
        self.actor_target = soft_update(self.actor_target, self.actor, self.tau)
        self.critic_target = soft_update(self.critic_target, self.critic, self.tau)

    @property
    def actor(self):
        return self.optim_actor.target

    @property
    def critic(self):
        return self.optim_critic.target

    def __str__(self):
        return "ddpg"
