from functools import partial

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network.actor import StateDependentGaussianPolicy
from rljax.network.critic import ContinuousQFunction
from rljax.utils import reparameterize


def build_sac_critic(state_space, action_space, hidden_units):
    def _func(x):
        return ContinuousQFunction(
            num_critics=2,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)

    fake_input = np.concatenate([state_space.sample(), action_space.sample()], axis=-1)
    return hk.transform(_func), fake_input[None, ...].astype(np.float32)


def build_sac_actor(state_space, action_space, hidden_units):
    def _func(x):
        return StateDependentGaussianPolicy(
            action_dim=action_space.shape[0],
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)

    fake_input = state_space.sample()
    return hk.transform(_func), fake_input[None, ...].astype(np.float32)


class SAC(OffPolicyActorCritic):
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
        lr_alpha=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
    ):
        super(SAC, self).__init__(
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
        self.critic, fake_input = build_sac_critic(state_space, action_space, units_critic)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), fake_input)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor, fake_input = build_sac_actor(state_space, action_space, units_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.actor.init(next(self.rng), fake_input)
        self.opt_state_actor = opt_init(self.params_actor)

        # Entropy coefficient.
        self.target_entropy = -float(action_space.shape[0])
        self.log_alpha = jnp.zeros((), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, None, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, log_std = self.actor.apply(params_actor, None, state)
        return reparameterize(mean, log_std, rng)[0]

    def update(self, writer):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, error = self._update_critic(
            opt_state_critic=self.opt_state_critic,
            params_critic=self.params_critic,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            rng=next(self.rng),
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(error)

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = self._update_actor(
            opt_state_actor=self.opt_state_actor,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            state=state,
            rng=next(self.rng),
        )

        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha = self._update_alpha(
            opt_state_alpha=self.opt_state_alpha,
            log_alpha=self.log_alpha,
            mean_log_pi=mean_log_pi,
        )

        # Update target network.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if self.learning_step % 1000 == 0:
            writer.add_scalar('loss/critic', loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
        rng: jnp.ndarray,
    ):
        (loss_critic, error), grad_critic = jax.value_and_grad(self._loss_critic, has_aux=True)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor=params_actor,
            log_alpha=log_alpha,
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
        params_actor: hk.Params,
        log_alpha: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
        rng: jnp.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        next_mean, next_log_std = self.actor.apply(params_actor, None, next_state)
        next_action, next_log_pi = reparameterize(next_mean, next_log_std, rng)
        next_q1, next_q2 = self.critic.apply(params_critic_target, None, jnp.concatenate([next_state, next_action], axis=1))
        next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_pi
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q1, curr_q2 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        error = jnp.abs(target_q - curr_q1)
        loss = (jnp.square(error) * weight).mean() + (jnp.square(target_q - curr_q2) * weight).mean()
        return loss, jax.lax.stop_gradient(error)

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: np.ndarray,
        state: np.ndarray,
        rng: np.ndarray,
    ):
        (loss_actor, mean_log_pi), grad_actor = jax.value_and_grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            state=state,
            rng=rng,
        )
        update, opt_state_actor = self.opt_actor(grad_actor, opt_state_actor)
        params_actor = optix.apply_updates(params_actor, update)
        return opt_state_actor, params_actor, loss_actor, mean_log_pi

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        rng: np.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        mean, log_std = self.actor.apply(params_actor, None, state)
        action, log_pi = reparameterize(mean, log_std, rng)
        q1, q2 = self.critic.apply(params_critic, None, jnp.concatenate([state, action], axis=1))
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - jnp.minimum(q1, q2).mean(), mean_log_pi

    @partial(jax.jit, static_argnums=0)
    def _update_alpha(
        self,
        opt_state_alpha,
        log_alpha: np.ndarray,
        mean_log_pi: np.ndarray,
    ):
        loss_alpha, grad_alpha = jax.value_and_grad(self._loss_alpha)(
            log_alpha,
            mean_log_pi=mean_log_pi,
        )
        update, opt_state_alpha = self.opt_alpha(grad_alpha, opt_state_alpha)
        log_alpha = optix.apply_updates(log_alpha, update)
        return opt_state_alpha, log_alpha, loss_alpha

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
        self,
        log_alpha: jnp.ndarray,
        mean_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_pi)

    def __str__(self):
        return "SAC"
