import numpy as np

import jax
import jax.numpy as jnp
from flax import optim
from haiku import PRNGSequence
from rljax.common.buffer import ReplayBuffer
from rljax.common.utils import soft_update
from rljax.sac.network import build_sac_actor, build_sac_critic, build_sac_log_alpha


@jax.jit
def update_actor_and_alpha(rng, optim_actor, optim_alpha, critic, state, target_entropy):
    actor, log_alpha = optim_actor.target, optim_alpha.target

    def actor_loss_fn(actor):
        action, log_pi = actor(state, key=rng, deterministic=False)
        q1, q2 = critic(state, action)
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha()))
        mean_log_pi = log_pi.mean()
        loss_actor = alpha * mean_log_pi - jnp.minimum(q1, q2).mean()
        return loss_actor, mean_log_pi

    grad_actor, mean_log_pi = jax.grad(actor_loss_fn, has_aux=True)(actor)

    def alpha_loss_fn(log_alpha):
        loss_alpha = -log_alpha() * (target_entropy + jax.lax.stop_gradient(mean_log_pi))
        return loss_alpha

    grad_alpha = jax.grad(alpha_loss_fn)(log_alpha)

    return optim_actor.apply_gradient(grad_actor), optim_alpha.apply_gradient(grad_alpha)


@jax.jit
def update_critic(rng, optim_critic, actor, critic_target, log_alpha, gamma, state, action, reward, done, next_state):
    critic = optim_critic.target

    next_action, next_log_pi = actor(next_state, key=rng, deterministic=False)
    next_q1, next_q2 = critic_target(next_state, next_action)
    next_q = jnp.minimum(next_q1, next_q2) - jnp.exp(log_alpha()) * next_log_pi
    target_q = jax.lax.stop_gradient(reward + (1.0 - done) * gamma * next_q)

    def critic_loss_fn(critic):
        curr_q1, curr_q2 = critic(state, action)
        loss_critic = jnp.square(target_q - curr_q1).mean() + jnp.square(target_q - curr_q2).mean()
        return loss_critic

    grad_critic = jax.grad(critic_loss_fn)(critic)
    return optim_critic.apply_gradient(grad_critic)


class SAC:
    def __init__(
        self,
        state_shape,
        action_shape,
        seed,
        gamma=0.99,
        batch_size=256,
        buffer_size=10 ** 6,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        start_steps=10000,
        tau=5e-3,
    ):
        # Initialize the sequence of random keys.
        self.rng = PRNGSequence(seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(buffer_size=buffer_size, state_shape=state_shape, action_shape=action_shape)

        # Actor.
        actor = build_sac_actor(
            state_shape=state_shape,
            action_shape=action_shape,
            rng_init=next(self.rng),
            hidden_units=units_actor,
        )
        self.optim_actor = jax.device_put(optim.Adam(lr_actor).create(actor))

        # Critic.
        critic = build_sac_critic(
            state_shape=state_shape,
            action_shape=action_shape,
            rng_init=next(self.rng),
            hidden_units=units_critic,
        )
        self.optim_critic = jax.device_put(optim.Adam(lr_critic).create(critic))

        # Target network.
        self.critic_target = build_sac_critic(
            state_shape=state_shape,
            action_shape=action_shape,
            rng_init=next(self.rng),
            hidden_units=units_critic,
        )

        # Entropy coefficient.
        log_alpha = build_sac_log_alpha(next(self.rng))
        self.optim_alpha = jax.device_put(optim.Adam(lr_alpha).create(log_alpha))
        self.target_entropy = -float(action_shape[0])

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def select_action(self, state):
        state = jax.device_put(state[None, ...])
        action = self.actor(state, deterministic=True)
        return np.array(action[0])

    def explore(self, state):
        state = jax.device_put(state[None, ...])
        action, _ = self.actor(state, key=next(self.rng), deterministic=False)
        return np.array(action[0])

    def step(self, env, state, t, step):
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)[0]

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        state, action, reward, done, next_state = self.buffer.sample(self.batch_size)

        self.optim_actor, self.optim_alpha = update_actor_and_alpha(
            rng=next(self.rng),
            optim_actor=self.optim_actor,
            optim_alpha=self.optim_alpha,
            critic=self.critic,
            state=state,
            target_entropy=self.target_entropy,
        )
        self.optim_critic = update_critic(
            rng=next(self.rng),
            optim_critic=self.optim_critic,
            actor=self.actor,
            critic_target=self.critic_target,
            log_alpha=self.log_alpha,
            gamma=self.gamma,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )
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
