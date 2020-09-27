from functools import partial

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network.actor import CategoricalPolicy
from rljax.network.critic import DiscreteQFunction
from rljax.utils import get_q_at_action


def build_sac_discrete_critic(action_dim, hidden_units, dueling_net):
    return hk.transform(
        lambda x: DiscreteQFunction(
            action_dim=action_dim,
            num_critics=2,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
            dueling_net=dueling_net,
        )(x)
    )


def build_sac_discrete_actor(action_dim, hidden_units):
    return hk.transform(
        lambda x: CategoricalPolicy(
            action_dim=action_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
        )(x)
    )


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
        batch_size=256,
        start_steps=1000,
        update_interval=1,
        tau=5e-3,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        units_actor=(512,),
        units_critic=(512,),
        target_entropy_ratio=0.8,
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
            tau=tau,
        )

        # Critic.
        fake_input = np.zeros((1, state_space.shape[0]), np.float32)
        self.critic = build_sac_discrete_critic(action_space.n, units_critic, dueling_net)
        opt_init_critic, self.opt_critic = optix.adam(lr_critic)
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), fake_input)
        self.opt_state_critic = opt_init_critic(self.params_critic)

        # Actor.
        self.actor = build_sac_discrete_actor(action_space.n, units_actor)
        opt_init_actor, self.opt_actor = optix.adam(lr_actor)
        self.params_actor = self.actor.init(next(self.rng), fake_input)
        self.opt_state_actor = opt_init_actor(self.params_actor)

        # Entropy coefficient.
        self.target_entropy = -np.log(1.0 / action_space.n) * target_entropy_ratio
        self.log_alpha = jnp.zeros((), dtype=jnp.float32)
        opt_init_alpha, self.opt_alpha = optix.adam(lr_alpha)
        self.opt_state_alpha = opt_init_alpha(self.log_alpha)

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0])

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

    def update(self):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
        self.opt_state_critic, self.params_critic, error = self._update_critic(
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

        # Update actor and alpha.
        self.opt_state_actor, self.opt_state_alpha, self.params_actor, self.log_alpha = self._update_actor_and_alpha(
            opt_state_actor=self.opt_state_actor,
            opt_state_alpha=self.opt_state_alpha,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            state=state,
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
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ):
        grad_critic, error = jax.grad(self._loss_critic, has_aux=True)(
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
        return opt_state_critic, params_critic, error

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
    ) -> jnp.ndarray:
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
    def _update_actor_and_alpha(
        self,
        opt_state_actor,
        opt_state_alpha,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
    ):
        grad_actor, mean_log_pi = jax.grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            state=state,
        )
        update, opt_state_actor = self.opt_actor(grad_actor, opt_state_actor)
        params_actor = optix.apply_updates(params_actor, update)

        grad_alpha = jax.grad(self._loss_alpha)(
            log_alpha,
            mean_log_pi=mean_log_pi,
        )
        update, opt_state_alpha = self.opt_alpha(grad_alpha, opt_state_alpha)
        log_alpha = optix.apply_updates(log_alpha, update)
        return opt_state_actor, opt_state_alpha, params_actor, log_alpha

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        curr_q1, curr_q2 = self.critic.apply(params_critic, None, state)
        curr_q = jax.lax.stop_gradient(jnp.minimum(curr_q1, curr_q2))
        pi, log_pi = self.actor.apply(params_actor, None, state)
        mean_log_pi = (pi * log_pi).sum(axis=1).mean()
        mean_q = (pi * curr_q).sum(axis=1).mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

    @partial(jax.jit, static_argnums=0)
    def _loss_alpha(
        self,
        log_alpha: jnp.ndarray,
        mean_log_pi: np.ndarray,
    ) -> jnp.ndarray:
        return -log_alpha * (self.target_entropy + mean_log_pi)

    def __str__(self):
        return "SAC-Discrete" if not self.use_per else "SAC-Discrete+PER"
