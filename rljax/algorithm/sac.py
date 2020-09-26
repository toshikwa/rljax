from functools import partial

import numpy as np

import flax
import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import ContinuousOffPolicyAlgorithm
from rljax.network.actor import StateDependentGaussianPolicy
from rljax.network.critic import ContinuousQFunction
from rljax.utils import reparameterize


class LogAlpha(flax.nn.Module):
    """
    Log of the entropy coefficient for SAC.
    """

    def apply(self):
        log_alpha = self.param("log_alpha", (), nn.initializers.zeros)
        return jnp.asarray(log_alpha, dtype=jnp.float32)


def build_sac_critic(action_dim, hidden_units):
    return hk.transform(
        lambda x: ContinuousQFunction(
            num_critics=2,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


def build_sac_actor(action_dim, hidden_units):
    return hk.transform(
        lambda x: StateDependentGaussianPolicy(
            action_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


def build_log_alpha(rng_init):
    _, param_init = LogAlpha.init(rng_init)
    return flax.nn.Model(LogAlpha, param_init)


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

        # Critic.
        self.critic = build_sac_critic(
            action_dim=action_space.shape[0],
            hidden_units=units_actor,
        )
        opt_init_critic, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(
            next(self.rng), np.zeros((1, state_space.shape[0] + action_space.shape[0]), np.float32)
        )
        self.opt_state_critic = opt_init_critic(self.params_critic)

        # Actor.
        self.actor = build_sac_actor(
            action_dim=action_space.shape[0],
            hidden_units=units_actor,
        )
        opt_init_actor, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.actor.init(next(self.rng), np.zeros((1, *state_space.shape), np.float32))
        self.opt_state_actor = opt_init_actor(self.params_actor)

        # Entropy coefficient.
        self.target_entropy = -float(action_space.shape[0])
        log_alpha = build_log_alpha(next(self.rng))
        self.opt_alpha = flax.optim.Adam(lr_alpha).create(log_alpha)

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(self, params_actor, state):
        mean, _ = self.actor.apply(params_actor, None, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(self, params_actor, rng, state):
        mean, log_std = self.actor.apply(params_actor, None, state)
        return reparameterize(mean, log_std, rng)[0]

    def update(self):
        self.learning_steps += 1
        _, (state, action, reward, done, next_state) = self.buffer.sample(self.batch_size)

        # Update critic.
        self.opt_state_critic, self.params_critic = self._update_critic(
            opt_state_critic=self.opt_state_critic,
            params_critic=self.params_critic,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.opt_alpha.target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            rng=next(self.rng),
        )

        # Update actor and alpha.
        self.opt_state_actor, self.params_actor, self.opt_alpha = self._update_actor_and_alpha(
            opt_state_actor=self.opt_state_actor,
            opt_alpha=self.opt_alpha,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            state=state,
            rng=next(self.rng),
        )

        # Update target network.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: flax.nn.Model,
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
            params_actor=params_actor,
            log_alpha=log_alpha,
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
        params_actor: hk.Params,
        log_alpha: flax.nn.Model,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        rng: jnp.ndarray,
    ) -> jnp.DeviceArray:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha()))
        next_mean, next_log_std = self.actor.apply(params_actor, None, next_state)
        next_action, next_log_pi = reparameterize(next_mean, next_log_std, rng)
        next_q1, next_q2 = self.critic.apply(params_critic_target, None, jnp.concatenate([next_state, next_action], axis=1))
        next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_pi
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

        curr_q1, curr_q2 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        return jnp.square(target_q - curr_q1).mean() + jnp.square(target_q - curr_q2).mean()

    @partial(jax.jit, static_argnums=0)
    def _update_actor_and_alpha(
        self,
        opt_state_actor,
        opt_alpha,
        params_actor: hk.Params,
        params_critic: hk.Params,
        state: np.ndarray,
        rng: np.ndarray,
    ):
        grad_actor, mean_log_pi = jax.grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=opt_alpha.target,
            state=state,
            rng=rng,
        )
        update, opt_state_actor = self.opt_actor(grad_actor, opt_state_actor)
        params_actor = optix.apply_updates(params_actor, update)

        grad_alpha = jax.grad(self._loss_alpha)(
            opt_alpha.target,
            mean_log_pi=mean_log_pi,
        )
        return opt_state_actor, params_actor, opt_alpha.apply_gradient(grad_alpha)

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: flax.nn.Model,
        state: np.ndarray,
        rng: np.ndarray,
    ) -> jnp.DeviceArray:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha()))
        mean, log_std = self.actor.apply(params_actor, None, state)
        action, log_pi = reparameterize(mean, log_std, rng)
        q1, q2 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - jnp.minimum(q1, q2).mean(), jax.lax.stop_gradient(mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
        self,
        log_alpha: flax.nn.Model,
        mean_log_pi: np.ndarray,
    ) -> jnp.DeviceArray:
        return -log_alpha() * (self.target_entropy + mean_log_pi)

    def __str__(self):
        return "sac"
