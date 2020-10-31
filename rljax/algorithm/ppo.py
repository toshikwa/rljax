from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base_class import OnPolicyActorCritic
from rljax.network import ContinuousVFunction, StateIndependentGaussianPolicy
from rljax.util import evaluate_gaussian_and_tanh_log_prob, optimize, reparameterize_gaussian_and_tanh


class PPO(OnPolicyActorCritic):
    name = "PPO"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=10.0,
        gamma=0.995,
        buffer_size=2048,
        batch_size=64,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        units_actor=(64, 64),
        units_critic=(64, 64),
        epoch_ppo=10,
        clip_eps=0.2,
        lambd=0.97,
    ):
        assert buffer_size % batch_size == 0
        super(PPO, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

        if fn_critic is None:

            def fn_critic(s):
                return ContinuousVFunction(
                    hidden_units=(units_critic),
                )(s)

        if fn_actor is None:

            def fn_actor(s):
                return StateIndependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                )(s)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(fn_critic))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(fn_actor))
        self.params_actor = self.params_actor_target = self.actor.init(next(self.rng), *self.fake_args_actor)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Other parameters.
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.max_grad_norm = max_grad_norm
        self.idxes = np.arange(buffer_size)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, state)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, state)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, True)

    def update(self, writer=None):
        state, action, reward, done, log_pi_old, next_state = self.buffer.get()

        # Calculate gamma-returns and GAEs.
        gae, target = self.calculate_gae(
            params_critic=self.params_critic,
            state=state,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        for _ in range(self.epoch_ppo):
            np.random.shuffle(self.idxes)
            for start in range(0, self.buffer_size, self.batch_size):
                self.learning_step += 1
                idx = self.idxes[start : start + self.batch_size]

                # Update critic.
                self.opt_state_critic, self.params_critic, loss_critic, _ = optimize(
                    self._loss_critic,
                    self.opt_critic,
                    self.opt_state_critic,
                    self.params_critic,
                    self.max_grad_norm,
                    state=state[idx],
                    target=target[idx],
                )

                # Update actor.
                self.opt_state_actor, self.params_actor, loss_actor, _ = optimize(
                    self._loss_actor,
                    self.opt_actor,
                    self.opt_state_actor,
                    self.params_actor,
                    self.max_grad_norm,
                    state=state[idx],
                    action=action[idx],
                    log_pi_old=log_pi_old[idx],
                    gae=gae[idx],
                )

        if writer:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        target: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.square(target - self.critic.apply(params_critic, state)).mean(), None

    @partial(jax.jit, static_argnums=0)
    def _loss_actor(
        self,
        params_actor: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        log_pi_old: np.ndarray,
        gae: jnp.ndarray,
    ) -> jnp.ndarray:
        # Calculate log(\pi) at current policy.
        mean, log_std = self.actor.apply(params_actor, state)
        log_pi = evaluate_gaussian_and_tanh_log_prob(mean, log_std, action)
        # Calculate importance ratio.
        ratio = jnp.exp(log_pi - log_pi_old)
        loss_actor1 = -ratio * gae
        loss_actor2 = -jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * gae
        loss_actor = jnp.maximum(loss_actor1, loss_actor2).mean()
        return loss_actor, None

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Current and next value estimates.
        value = jax.lax.stop_gradient(self.critic.apply(params_critic, state))
        next_value = jax.lax.stop_gradient(self.critic.apply(params_critic, next_state))
        # Calculate TD errors.
        delta = reward + self.gamma * next_value * (1.0 - done) - value
        # Calculate GAE recursively from behind.
        gae = [delta[-1]]
        for t in jnp.arange(self.buffer_size - 2, -1, -1):
            gae.insert(0, delta[t] + self.gamma * self.lambd * (1 - done[t]) * gae[0])
        gae = jnp.array(gae)
        return (gae - gae.mean()) / (gae.std() + 1e-8), gae + value
