from functools import partial
from typing import Any, Tuple

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network.actor import CategoricalPolicy
from rljax.network.critic import DiscreteQFunction, DQNBody
from rljax.utils import get_q_at_action


def build_sac_discrete_critic(state_space, action_space, hidden_units, dueling_net):
    def _func(x):
        if len(state_space.shape) == 3:
            return [
                DiscreteQFunction(
                    action_dim=action_space.n,
                    num_critics=1,
                    hidden_units=hidden_units,
                    hidden_activation=nn.relu,
                    dueling_net=dueling_net,
                )(DQNBody()(x))
                for _ in range(2)
            ]
        elif len(state_space.shape) == 1:
            return DiscreteQFunction(
                action_dim=action_space.n,
                num_critics=2,
                hidden_units=hidden_units,
                hidden_activation=nn.relu,
                dueling_net=dueling_net,
            )(x)

    fake_input = state_space.sample()
    if len(state_space.shape) == 1:
        fake_input = fake_input.astype(np.float32)
    return hk.transform(_func), fake_input[None, ...].astype(np.float32)


def build_sac_discrete_actor(state_space, action_space, hidden_units):
    def _func(x):
        if len(state_space.shape) == 3:
            x = DQNBody()(x)
        return CategoricalPolicy(
            action_dim=action_space.n,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)

    fake_input = state_space.sample()
    return hk.transform(_func), fake_input[None, ...].astype(np.float32)


class SACDiscrete(OffPolicyActorCritic):
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
        batch_size=64,
        start_steps=20000,
        update_interval=4,
        update_interval_target=8000,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(512,),
        units_critic=(512,),
        target_entropy_ratio=0.98,
        dueling_net=True,
    ):
        super(SACDiscrete, self).__init__(
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
            update_interval_target=update_interval_target,
        )

        # Critic.
        self.critic, fake_input = build_sac_discrete_critic(state_space, action_space, units_critic, dueling_net)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), fake_input)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor, fake_input = build_sac_discrete_actor(state_space, action_space, units_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.actor.init(next(self.rng), fake_input)
        self.opt_state_actor = opt_init(self.params_actor)

        # Entropy coefficient.
        self.target_entropy = -np.log(1.0 / action_space.n) * target_entropy_ratio
        self.log_alpha = jnp.zeros((), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        pi, _ = self.actor.apply(params_actor, None, state)
        return jnp.argmax(pi, axis=1)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        rng: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        pi, _ = self.actor.apply(params_actor, None, state)
        return jax.random.categorical(rng, pi)

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
        )

        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha = self._update_alpha(
            opt_state_alpha=self.opt_state_alpha,
            log_alpha=self.log_alpha,
            mean_log_pi=mean_log_pi,
        )

        # Update target network.
        if self.env_step % self.update_interval_target == 0:
            self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic: Any,
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
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
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
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.exp(log_alpha)
        pi, log_pi = self.actor.apply(params_actor, None, next_state)
        next_q1, next_q2 = self.critic.apply(params_critic_target, None, next_state)
        next_q = (pi * (jnp.minimum(next_q1, next_q2) - alpha * log_pi)).sum(axis=1, keepdims=True)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q_s1, curr_q_s2 = self.critic.apply(params_critic, None, state)
        curr_q1, curr_q2 = get_q_at_action(curr_q_s1, action), get_q_at_action(curr_q_s2, action)
        error = jnp.abs(target_q - curr_q1)
        loss = (jnp.square(error) * weight).mean() + (jnp.square(target_q - curr_q2) * weight).mean()
        return loss, jax.lax.stop_gradient(error)

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor: Any,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_actor, mean_log_pi), grad_actor = jax.value_and_grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            state=state,
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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.exp(log_alpha)
        curr_q1, curr_q2 = self.critic.apply(params_critic, None, state)
        curr_q = jax.lax.stop_gradient(jnp.minimum(curr_q1, curr_q2))
        pi, log_pi = self.actor.apply(params_actor, None, state)
        mean_log_pi = (pi * log_pi).sum(axis=1).mean()
        mean_q = (pi * curr_q).sum(axis=1).mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _update_alpha(
        self,
        opt_state_alpha: Any,
        log_alpha: jnp.ndarray,
        mean_log_pi: jnp.ndarray,
    ) -> Tuple[Any, jnp.ndarray, jnp.ndarray]:
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
        return "SAC-Discrete" if not self.use_per else "SAC-Discrete+PER"
