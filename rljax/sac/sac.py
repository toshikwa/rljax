from functools import partial
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
from flax import nn, optim
from rljax.common.base_class import ContinuousOffPolicyAlgorithm
from rljax.common.utils import soft_update, update_network
from rljax.sac.network import build_sac_actor, build_sac_critic, build_sac_log_alpha


def critic_grad_fn(
    rng: np.ndarray,
    actor: nn.Model,
    critic: nn.Model,
    critic_target: nn.Model,
    log_alpha: nn.Model,
    discount: float,
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    next_state: jnp.ndarray,
) -> nn.Model:
    alpha = jax.lax.stop_gradient(jnp.exp(log_alpha()))
    next_action, next_log_pi = actor(next_state, key=rng, deterministic=False)
    next_q1, next_q2 = critic_target(next_state, next_action)
    next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_pi
    target_q = jax.lax.stop_gradient(reward + (1.0 - done) * discount * next_q)

    def critic_loss_fn(critic):
        curr_q1, curr_q2 = critic(state, action)
        loss_critic = jnp.square(target_q - curr_q1).mean() + jnp.square(target_q - curr_q2).mean()
        return loss_critic

    grad_critic = jax.grad(critic_loss_fn)(critic)
    return grad_critic


def actor_and_alpha_grad_fn(
    rng: np.ndarray,
    actor: nn.Model,
    critic: nn.Model,
    log_alpha: nn.Model,
    target_entropy: float,
    state: jnp.ndarray,
) -> Tuple[nn.Model, nn.Model]:
    alpha = jax.lax.stop_gradient(jnp.exp(log_alpha()))

    def actor_loss_fn(actor):
        action, log_pi = actor(state, key=rng, deterministic=False)
        q1, q2 = critic(state, action)
        mean_log_pi = log_pi.mean()
        loss_actor = alpha * mean_log_pi - jnp.minimum(q1, q2).mean()
        return loss_actor, mean_log_pi

    grad_actor, mean_log_pi = jax.grad(actor_loss_fn, has_aux=True)(actor)
    mean_log_pi = jax.lax.stop_gradient(mean_log_pi)

    def alpha_loss_fn(log_alpha):
        loss_alpha = -log_alpha() * (target_entropy + mean_log_pi)
        return loss_alpha

    grad_alpha = jax.grad(alpha_loss_fn)(log_alpha)
    return grad_actor, grad_alpha


class SAC(ContinuousOffPolicyAlgorithm):
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
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
    ):
        super(SAC, self).__init__(
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
        actor = build_sac_actor(
            state_dim=state_space.shape[0],
            action_dim=action_space.shape[0],
            rng_init=next(self.rng),
            hidden_units=units_actor,
        )
        self.optim_actor = jax.device_put(optim.Adam(learning_rate=lr_actor).create(actor))

        # Critic.
        rng_critic = next(self.rng)
        critic = build_sac_critic(
            state_dim=state_space.shape[0],
            action_dim=action_space.shape[0],
            rng_init=rng_critic,
            hidden_units=units_critic,
        )
        self.optim_critic = jax.device_put(optim.Adam(learning_rate=lr_critic).create(critic))

        # Target network.
        self.critic_target = jax.device_put(
            build_sac_critic(
                state_dim=state_space.shape[0],
                action_dim=action_space.shape[0],
                rng_init=rng_critic,
                hidden_units=units_critic,
            )
        )

        # Entropy coefficient.
        target_entropy = -float(action_space.shape[0])
        log_alpha = build_sac_log_alpha(next(self.rng))
        self.optim_alpha = jax.device_put(optim.Adam(learning_rate=lr_alpha).create(log_alpha))

        # Compile functions.
        self.critic_grad_fn = jax.jit(partial(critic_grad_fn, discount=self.discount))
        self.actor_and_alpha_grad_fn = jax.jit(partial(actor_and_alpha_grad_fn, target_entropy=target_entropy))

    def select_action(self, state):
        state = jax.device_put(state[None, ...])
        action = self.actor(state, deterministic=True)
        return np.array(action[0])

    def explore(self, state):
        state = jax.device_put(state[None, ...])
        action, _ = self.actor(state, key=next(self.rng), deterministic=False)
        return np.array(action[0])

    def update(self):
        self.learning_steps += 1
        _, (state, action, reward, done, next_state) = self.buffer.sample(self.batch_size)

        # Update critic.
        grad_critic = self.critic_grad_fn(
            rng=next(self.rng),
            actor=self.actor,
            critic=self.critic,
            critic_target=self.critic_target,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )
        self.optim_critic = update_network(self.optim_critic, grad_critic)

        # Update actor and log alpha.
        grad_actor, grad_alpha = self.actor_and_alpha_grad_fn(
            rng=next(self.rng),
            actor=self.actor,
            critic=self.critic,
            log_alpha=self.log_alpha,
            state=state,
        )
        self.optim_actor = update_network(self.optim_actor, grad_actor)
        self.optim_alpha = update_network(self.optim_alpha, grad_alpha)

        # Update target network.
        self.critic_target = soft_update(self.critic_target, self.critic, self.tau)

    @property
    def actor(self):
        return self.optim_actor.target

    @property
    def critic(self):
        return self.optim_critic.target

    @property
    def log_alpha(self):
        return self.optim_alpha.target

    def __str__(self):
        return "sac"
