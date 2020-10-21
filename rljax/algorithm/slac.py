import math
import os
from functools import partial
from typing import Any, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import Algorithm
from rljax.buffer import SLACReplayBuffer
from rljax.network import (
    ConstantGaussian,
    ContinuousQFunction,
    Gaussian,
    SLACDecoder,
    SLACEncoder,
    StateDependentGaussianPolicy,
)
from rljax.util import calculate_kl_divergence, load_params, reparameterize_gaussian_and_tanh, save_params, soft_update


class SLAC(Algorithm):
    name = "SLAC"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        buffer_size=10 ** 5,
        batch_size_sac=256,
        batch_size_latent=32,
        start_steps=10000,
        initial_learning_steps=10 ** 5,
        update_interval=1,
        tau=5e-3,
        num_sequences=8,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        lr_latent=1e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_latent=(256, 256),
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
    ):
        assert len(state_space.shape) == 3 and state_space.shape[:2] == (64, 64)
        super(SLAC, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
        )

        self.buffer = SLACReplayBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            num_sequences=num_sequences,
        )

        fake_state_ = jnp.empty((1, num_sequences, *state_space.shape), dtype=jnp.uint8)
        fake_action_ = jnp.empty((1, num_sequences, *action_space.shape))
        fake_feature = jnp.empty((1, feature_dim))
        fake_feature_action = jnp.empty((1, num_sequences * feature_dim + (num_sequences - 1) * action_space.shape[0]))
        fake_z = jnp.empty((1, z1_dim + z2_dim))
        fake_z1 = jnp.empty((1, z1_dim))
        fake_z2 = jnp.empty((1, z2_dim))
        fake_z_ = jnp.empty((1, num_sequences, z1_dim + z2_dim))
        fake_z1_ = jnp.empty((1, num_sequences, z1_dim))
        fake_z2_ = jnp.empty((1, num_sequences, z2_dim))

        def z1_prior_fn(z2, a):
            return Gaussian(output_dim=z1_dim, hidden_units=units_latent)(jnp.concatenate([z2, a], axis=1))

        def z1_post_fn(f, z2, a):
            return Gaussian(output_dim=z1_dim, hidden_units=units_latent)(jnp.concatenate([f, z2, a], axis=1))

        def z2_fn(z1, z2, a):
            return Gaussian(output_dim=z2_dim, hidden_units=units_latent)(jnp.concatenate([z1, z2, a], axis=1))

        def reward_fn(z_, a_, n_z_):
            x = jnp.concatenate([z_, a_, n_z_], axis=-1)
            B, S, X = x.shape
            mean, std = Gaussian(output_dim=1, hidden_units=units_latent)(x.reshape([B * S, X]))
            return mean.reshape([B, S, 1]), std.reshape([B, S, 1])

        def encoder_fn(x):
            return SLACEncoder(output_dim=feature_dim)(x)

        def decoder_fn(z1_, z2_):
            return SLACDecoder(state_space=state_space, std=np.sqrt(0.1))(jnp.concatenate([z1_, z2_], axis=-1))

        # p(z1(0)) = N(0, I)
        self.z1_prior_init = hk.without_apply_rng(hk.transform(lambda x: ConstantGaussian(z1_dim, 1.0)(x)))
        params_z1_prior_init = self.z1_prior_init.init(next(self.rng), self.fake_action)
        # p(z1(t+1) | z2(t), a(t))
        self.z1_prior = hk.without_apply_rng(hk.transform(z1_prior_fn))
        params_z1_prior = self.z1_prior.init(next(self.rng), fake_z2, self.fake_action)

        # q(z1(0) | feat(0))
        self.z1_post_init = hk.without_apply_rng(hk.transform(lambda x: Gaussian(z1_dim, units_latent)(x)))
        params_z1_post_init = self.z1_post_init.init(next(self.rng), fake_feature)
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        self.z1_post = hk.without_apply_rng(hk.transform(z1_post_fn))
        params_z1_post = self.z1_post.init(next(self.rng), fake_feature, fake_z2, self.fake_action)

        # p(z2(0) | z1(0)) == q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        self.z2_init = hk.without_apply_rng(hk.transform(lambda x: Gaussian(z2_dim, units_latent)(x)))
        params_z2_init = self.z2_init.init(next(self.rng), fake_z1)
        # p(z2(t+1) | z1(t+1), z2(t), a(t)) == q(z2(t+1) | z1(t+1), z2(t), a(t))
        self.z2 = hk.without_apply_rng(hk.transform(z2_fn))
        params_z2 = self.z2.init(next(self.rng), fake_z1, fake_z2, self.fake_action)

        # p(r(t) | z1(t), z2(t), a(t), z1(t+1), z2(t+1))
        self.reward = hk.without_apply_rng(hk.transform(reward_fn))
        params_reward = self.reward.init(next(self.rng), fake_z_, fake_action_, fake_z_)

        # feature(t) = f(x(t))
        self.encoder = hk.without_apply_rng(hk.transform(encoder_fn))
        params_encoder = self.encoder.init(next(self.rng), fake_state_)
        # p(x(t) | z1(t), z2(t))
        self.decoder = hk.without_apply_rng(hk.transform(decoder_fn))
        params_decoder = self.decoder.init(next(self.rng), fake_z1_, fake_z2_)

        # Stochastic Latent Variable Model.
        self.params_latent = hk.data_structures.to_immutable_dict(
            {
                "z1_prior_init": params_z1_prior_init,
                "z1_prior": params_z1_prior,
                "z1_post_init": params_z1_post_init,
                "z1_post": params_z1_post,
                "z2_init": params_z2_init,
                "z2": params_z2,
                "reward": params_reward,
                "encoder": params_encoder,
                "decoder": params_decoder,
            }
        )
        opt_init, self.opt_latent = optix.adam(lr_latent)
        self.opt_state_latent = opt_init(self.params_latent)

        def critic_fn(z, a):
            return ContinuousQFunction(
                num_critics=2,
                hidden_units=units_critic,
            )(z, a)

        def actor_fn(x):
            return StateDependentGaussianPolicy(
                action_space=action_space,
                hidden_units=units_actor,
                clip_log_std=True,
            )(x)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        self.params_critic = self.params_critic_target = self.critic.init(next(self.rng), fake_z, self.fake_action)
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor.
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.params_actor = self.actor.init(next(self.rng), fake_feature_action)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Entropy coefficient.
        self.target_entropy = -float(action_space.shape[0])
        self.log_alpha = jnp.zeros((), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha)
        self.opt_state_alpha = opt_init(self.log_alpha)

        # Other parameters.
        self.learning_step_latent = 0
        self.learning_step_sac = 0
        self.discount = gamma
        self.batch_size_sac = batch_size_sac
        self.batch_size_latent = batch_size_latent
        self.start_steps = start_steps
        self.initial_learning_steps = initial_learning_steps
        self.num_sequences = num_sequences
        self.update_interval = update_interval
        self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self):
        return self.agent_step % self.update_interval == 0 and self.agent_step >= self.start_steps

    def step(self, env, input):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(input)

        state, reward, done, _ = env.step(action)
        mask = False if self.episode_step == env._max_episode_steps else done
        input.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            self.episode_step = 0
            state = env.reset()
            input.reset_episode(state)
            self.buffer.reset_episode(state)

        return None

    def select_action(self, input):
        feature_action = self._preprocess(self.params_latent["encoder"], input.state, input.action)
        action = self._select_action(self.params_actor, feature_action)
        return np.array(action[0])

    def explore(self, input):
        feature_action = self._preprocess(self.params_latent["encoder"], input.state, input.action)
        action = self._explore(self.params_actor, next(self.rng), feature_action)
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _preprocess(
        self,
        params_encoder: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        feature = self.encoder.apply(params_encoder, state).reshape([1, -1])
        return jnp.concatenate([feature, action], axis=1)

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_actor: hk.Params,
        feature_action: np.ndarray,
    ) -> jnp.ndarray:
        mean, _ = self.actor.apply(params_actor, feature_action)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_actor: hk.Params,
        key: jnp.ndarray,
        feature_action: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        mean, log_std = self.actor.apply(params_actor, feature_action)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    @partial(jax.jit, static_argnums=0)
    def get_latent_batch(
        self,
        params_latent: hk.Params,
        state_: np.ndarray,
        action_: np.ndarray,
        keys: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        N = state_.shape[0]
        # feature(1:t+1)
        feature_ = self.encoder.apply(params_latent["encoder"], state_)
        # z(1:t+1)
        z_ = jax.lax.stop_gradient(jnp.concatenate(self.sample_post(params_latent, feature_, action_, keys)[2:], axis=-1))
        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1))
        feature_action = jnp.concatenate([feature_[:, :-1].reshape([N, -1]), action_[:, :-1].reshape([N, -1])], axis=-1)
        # fa(t+1)=(x(2:t+1), a(2:t))
        next_feature_action = jnp.concatenate([feature_[:, 1:].reshape([N, -1]), action_[:, 1:].reshape([N, -1])], axis=-1)
        return z, next_z, action, feature_action, next_feature_action

    def update_sac(self, writer=None):
        self.learning_step_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.get_latent_batch(
            params_latent=self.params_latent,
            state_=state_,
            action_=action_,
            keys=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
        )

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic = self._update_critic(
            opt_state_critic=self.opt_state_critic,
            params_critic=self.params_critic,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            z=z,
            next_z=next_z,
            action=action,
            reward=reward,
            done=done,
            next_feature_action=next_feature_action,
            key=next(self.rng),
        )

        # Update actor and alpha.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = self._update_actor(
            opt_state_actor=self.opt_state_actor,
            params_actor=self.params_actor,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            z=z,
            feature_action=feature_action,
            key=next(self.rng),
        )
        self.opt_state_alpha, self.log_alpha, loss_alpha = self._update_alpha(
            opt_state_alpha=self.opt_state_alpha,
            log_alpha=self.log_alpha,
            mean_log_pi=mean_log_pi,
        )

        # Update target network.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if writer and self.learning_step_sac % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step_sac)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step_sac)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step_sac)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step_sac)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step_sac)

    def update_latent(self, writer=None):
        self.learning_step_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)

        self.opt_state_latent, self.params_latent, loss_latent = self._update_latent(
            opt_state_latent=self.opt_state_latent,
            params_latent=self.params_latent,
            state_=state_,
            action_=action_,
            reward_=reward_,
            done_=done_,
            keys1=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
            keys2=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
        )

        if writer and self.learning_step_latent % 1000 == 0:
            writer.add_scalar("loss/latent", loss_latent, self.learning_step_latent)

    @partial(jax.jit, static_argnums=0)
    def _update_critic(
        self,
        opt_state_critic: Any,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        z: np.ndarray,
        next_z: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_feature_action: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray]:
        loss_critic, grad_critic = jax.value_and_grad(self._loss_critic)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor=params_actor,
            log_alpha=log_alpha,
            z=z,
            next_z=next_z,
            action=action,
            reward=reward,
            done=done,
            next_feature_action=next_feature_action,
            key=key,
        )
        update, opt_state_critic = self.opt_critic(grad_critic, opt_state_critic)
        params_critic = optix.apply_updates(params_critic, update)
        return opt_state_critic, params_critic, loss_critic

    @partial(jax.jit, static_argnums=0)
    def _loss_critic(
        self,
        params_critic: hk.Params,
        params_critic_target: hk.Params,
        params_actor: hk.Params,
        log_alpha: jnp.ndarray,
        z: np.ndarray,
        next_z: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_feature_action: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.exp(log_alpha)
        # Sample next actions.
        next_mean, next_log_std = self.actor.apply(params_actor, next_feature_action)
        next_action, next_log_pi = reparameterize_gaussian_and_tanh(next_mean, next_log_std, key, True)
        # Calculate target soft q values (clipped double q) with target critic.
        next_q1, next_q2 = self.critic.apply(params_critic_target, next_z, next_action)
        next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_pi
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        # Calculate current soft q values with online critic.
        curr_q1, curr_q2 = self.critic.apply(params_critic, z, action)
        return jnp.square(target_q - curr_q1).mean() + jnp.square(target_q - curr_q2).mean()

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor: Any,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        z: np.ndarray,
        feature_action: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_actor, mean_log_pi), grad_actor = jax.value_and_grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            z=z,
            feature_action=feature_action,
            key=key,
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
        z: np.ndarray,
        feature_action: np.ndarray,
        key: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        # Sample actions.
        mean, log_std = self.actor.apply(params_actor, feature_action)
        action, log_pi = reparameterize_gaussian_and_tanh(mean, log_std, key, True)
        # Calculate soft q values with online critic.
        q1, q2 = self.critic.apply(params_critic, z, action)
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - jnp.minimum(q1, q2).mean(), jax.lax.stop_gradient(mean_log_pi)

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

    @partial(jax.jit, static_argnums=0)
    def _update_latent(
        self,
        opt_state_latent: Any,
        params_latent: hk.Params,
        state_: np.ndarray,
        action_: np.ndarray,
        reward_: np.ndarray,
        done_: np.ndarray,
        keys1: List[np.ndarray],
        keys2: List[np.ndarray],
    ) -> Tuple[Any, hk.Params, jnp.ndarray]:
        loss_latent, grad_latent = jax.value_and_grad(self._loss_latent)(
            params_latent,
            state_=state_,
            action_=action_,
            reward_=reward_,
            done_=done_,
            keys1=keys1,
            keys2=keys2,
        )
        update, opt_state_latent = self.opt_latent(grad_latent, opt_state_latent)
        params_latent = optix.apply_updates(params_latent, update)
        return opt_state_latent, params_latent, loss_latent

    @partial(jax.jit, static_argnums=0)
    def _loss_latent(
        self,
        params_latent: hk.Params,
        state_: np.ndarray,
        action_: np.ndarray,
        reward_: np.ndarray,
        done_: np.ndarray,
        keys1: List[np.ndarray],
        keys2: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate the sequence of features.
        feature_ = self.encoder.apply(params_latent["encoder"], state_)

        # Sample from stochastic latent variable model.
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(params_latent, action_, keys1)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_post(params_latent, feature_, action_, keys2)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(axis=0).sum()

        # Prediction loss of images.
        state_mean_, state_std_ = self.decoder.apply(params_latent["decoder"], z1_, z2_)
        log_likelihood_ = -0.5 * (jnp.square((state_ - state_mean_) / state_std_) + 2 * jnp.log(state_std_))
        loss_image = -log_likelihood_.mean(axis=0).sum()

        # Prediction loss of rewards.
        z_ = jnp.concatenate([z1_, z2_], axis=-1)
        reward_mean_, reward_std_ = self.reward.apply(params_latent["reward"], z_[:, :-1], action_, z_[:, 1:])
        log_likelihood_reward_ = -0.5 * (jnp.square((reward_ - reward_mean_) / reward_std_) + 2 * jnp.log(reward_std_))
        loss_reward = -(log_likelihood_reward_ * (1 - done_)).mean(axis=0).sum()

        return loss_kld + loss_image + loss_reward

    @partial(jax.jit, static_argnums=0)
    def sample_prior(
        self,
        params_latent: hk.Params,
        action_: jnp.ndarray,
        keys: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z1_mean_ = []
        z1_std_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.z1_prior_init.apply(params_latent["z1_prior_init"], action_[:, 0])
        z1 = z1_mean + jax.random.normal(keys[0], z1_std.shape) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_init.apply(params_latent["z2_init"], z1)
        z2 = z2_mean + jax.random.normal(keys[1], z2_std.shape) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)

        for t in range(1, action_.shape[1] + 1):
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean, z1_std = self.z1_prior.apply(params_latent["z1_prior"], z2, action_[:, t - 1])
            z1 = z1_mean + jax.random.normal(keys[2 * t], z1_std.shape) * z1_std
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.z2.apply(params_latent["z2"], z1, z2, action_[:, t - 1])
            z2 = z2_mean + jax.random.normal(keys[2 * t + 1], z2_std.shape) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)

        z1_mean_ = jnp.stack(z1_mean_, axis=1)
        z1_std_ = jnp.stack(z1_std_, axis=1)
        return (z1_mean_, z1_std_)

    @partial(jax.jit, static_argnums=0)
    def sample_post(
        self,
        params_latent: hk.Params,
        feature_: jnp.ndarray,
        action_: jnp.ndarray,
        keys: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z1_mean_ = []
        z1_std_ = []
        z1_ = []
        z2_ = []

        # q(z1(0) | feat(0))
        z1_mean, z1_std = self.z1_post_init.apply(params_latent["z1_post_init"], feature_[:, 0])
        z1 = z1_mean + jax.random.normal(keys[0], z1_std.shape) * z1_std
        # q(z2(0) | z1(0))
        z2_mean, z2_std = self.z2_init.apply(params_latent["z2_init"], z1)
        z2 = z2_mean + jax.random.normal(keys[1], z2_std.shape) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        for t in range(1, action_.shape[1] + 1):
            # q(z1(t+1) | feat(t+1), z2(t), a(t))
            z1_mean, z1_std = self.z1_post.apply(params_latent["z1_post"], feature_[:, t], z2, action_[:, t - 1])
            z1 = z1_mean + jax.random.normal(keys[2 * t], z1_std.shape) * z1_std
            # q(z2(t+1) | z1(t+1), z2(t), a(t))
            z2_mean, z2_std = self.z2.apply(params_latent["z2"], z1, z2, action_[:, t - 1])
            z2 = z2_mean + jax.random.normal(keys[2 * t + 1], z2_std.shape) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = jnp.stack(z1_mean_, axis=1)
        z1_std_ = jnp.stack(z1_std_, axis=1)
        z1_ = jnp.stack(z1_, axis=1)
        z2_ = jnp.stack(z2_, axis=1)
        return (z1_mean_, z1_std_, z1_, z2_)

    def save_params(self, save_dir):
        super(SLAC, self).save_params(save_dir)
        save_params(self.params_latent, os.path.join(save_dir, "params_latent.npz"))
        save_params(self.params_critic, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_latent = load_params(os.path.join(save_dir, "params_latent.npz"))
        self.params_critic = self.params_critic_target = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = load_params(os.path.join(save_dir, "params_actor.npz"))

    def update(self, writer):
        NotImplementedError
