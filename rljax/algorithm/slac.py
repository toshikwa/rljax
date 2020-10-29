import os
from collections import deque
from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import SlacAlgorithm
from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQFunction, StateDependentGaussianPolicy, make_stochastic_latent_variable_model
from rljax.util import calculate_kl_divergence, gaussian_log_prob, load_params, optimize, save_params


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_space, action_space, num_sequences):
        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state, dtype=np.uint8)[None, ...]

    @property
    def action(self):
        return np.array(self._action, dtype=np.float32).reshape(1, -1)


class SLAC(SlacAlgorithm, SAC):
    name = "SLAC"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        num_sequences=8,
        buffer_size=50000,
        batch_size_sac=256,
        batch_size_model=32,
        start_steps=10000,
        initial_learning_steps=50,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        lr_model=1e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_model=(256, 256),
        d2rl=False,
        init_alpha=1.0,
        adam_b1_alpha=0.9,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
    ):
        assert len(state_space.shape) == 3 and state_space.shape[:2] == (64, 64)
        SlacAlgorithm.__init__(
            self,
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            num_sequences=num_sequences,
            buffer_size=buffer_size,
            batch_size_sac=batch_size_sac,
            batch_size_model=batch_size_model,
            start_steps=start_steps,
            initial_learning_steps=initial_learning_steps,
            update_interval=update_interval,
            tau=tau,
        )
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(z, a):
                return ContinuousQFunction(
                    num_critics=2,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(z, a)

        if fn_actor is None:

            def fn_actor(x):
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    d2rl=d2rl,
                )(x)

        fake_z = jnp.empty((1, z1_dim + z2_dim))
        fake_action = action_space.sample().astype(np.float32)[None, ...]
        fake_feature_action = jnp.empty((1, num_sequences * feature_dim + (num_sequences - 1) * action_space.shape[0]))
        self.fake_args_critic = (fake_z, fake_action)
        self.fake_args_actor = (fake_feature_action,)
        self.setup_soft_actor_critic(fn_actor, fn_critic, lr_actor, lr_critic, lr_alpha, init_alpha, adam_b1_alpha)

        self.model, self.params_model = make_stochastic_latent_variable_model(
            rng=self.rng,
            state_space=state_space,
            action_space=action_space,
            num_sequences=num_sequences,
            units_model=units_model,
            z1_dim=z1_dim,
            z2_dim=z2_dim,
            feature_dim=feature_dim,
        )
        opt_init, self.opt_model = optix.adam(lr_model)
        self.opt_state_model = opt_init(self.params_model)

    @partial(jax.jit, static_argnums=0)
    def _preprocess(
        self,
        params_model: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        feature = self.model["encoder"].apply(params_model["encoder"], state).reshape([1, -1])
        return jnp.concatenate([feature, action], axis=1)

    @partial(jax.jit, static_argnums=0)
    def get_input_for_sac(
        self,
        params_model: hk.Params,
        state_: np.ndarray,
        action_: np.ndarray,
        key_list: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        N = state_.shape[0]
        feature_ = self.model["encoder"].apply(params_model["encoder"], state_)
        z_ = jax.lax.stop_gradient(jnp.concatenate(self.sample_post(params_model, feature_, action_, key_list)[2:], axis=-1))
        z, next_z = z_[:, -2], z_[:, -1]
        action = action_[:, -1]
        feature_action = jnp.concatenate([feature_[:, :-1].reshape([N, -1]), action_[:, :-1].reshape([N, -1])], axis=-1)
        next_feature_action = jnp.concatenate([feature_[:, 1:].reshape([N, -1]), action_[:, 1:].reshape([N, -1])], axis=-1)
        return z, next_z, action, feature_action, next_feature_action

    def update_sac(self, writer=None):
        self.learning_step_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.get_input_for_sac(
            params_model=self.params_model,
            state_=state_,
            action_=action_,
            key_list=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
        )

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, _ = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
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
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
            self._loss_actor,
            self.opt_actor,
            self.opt_state_actor,
            self.params_actor,
            self.max_grad_norm,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            z=z,
            feature_action=feature_action,
            key=next(self.rng),
        )
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            None,
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

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_z: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        next_q_list = self.critic.apply(params_critic_target, next_z, next_action)
        next_q = jnp.asarray(next_q_list).min(axis=0) - alpha * next_log_pi
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

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
        next_action, next_log_pi = self._sample_action(params_actor, key, next_feature_action)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_z, next_action, next_log_pi)
        curr_q_list = self.critic.apply(params_critic, z, action)
        loss = 0.0
        for curr_q in curr_q_list:
            loss += jnp.square(target - curr_q).mean()
        return loss, None

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
        action, log_pi = self._sample_action(params_actor, key, feature_action)
        mean_q = jnp.asarray(self.critic.apply(params_critic, z, action)).min(axis=0).mean()
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

    def update_model(self, writer=None):
        self.learning_step_model += 1
        state_, action_, reward_, done_ = self.buffer.sample_model(self.batch_size_model)

        self.opt_state_model, self.params_model, loss_model, _ = optimize(
            self._loss_model,
            self.opt_model,
            self.opt_state_model,
            self.params_model,
            self.max_grad_norm,
            state_=state_,
            action_=action_,
            reward_=reward_,
            done_=done_,
            key_list1=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
            key_list2=[next(self.rng) for _ in range(2 * (self.num_sequences + 1))],
        )

        if writer and self.learning_step_model % 1000 == 0:
            writer.add_scalar("loss/latent", loss_model, self.learning_step_model)

    @partial(jax.jit, static_argnums=0)
    def _loss_model(
        self,
        params_model: hk.Params,
        state_: np.ndarray,
        action_: np.ndarray,
        reward_: np.ndarray,
        done_: np.ndarray,
        key_list1: List[np.ndarray],
        key_list2: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Floatify the sequence of images.
        img_ = state_.astype(jnp.float32) / 255.0
        # Calculate the sequence of features.
        feature_ = self.model["encoder"].apply(params_model["encoder"], state_)

        # Sample from stochastic latent variable model.
        z1_mean_pri_, z1_std_pri_ = self.sample_prior(params_model, action_, key_list1)
        z1_mean_post_, z1_std_post_, z1_, z2_ = self.sample_post(params_model, feature_, action_, key_list2)

        # Calculate KL divergence loss.
        loss_kld = calculate_kl_divergence(z1_mean_post_, z1_std_post_, z1_mean_pri_, z1_std_pri_).mean(axis=0).sum()
        # Prediction loss of images.
        img_mean_, img_std_ = self.model["decoder"].apply(params_model["decoder"], z1_, z2_)
        loss_img = -gaussian_log_prob(jnp.log(img_std_), (img_ - img_mean_) / img_std_).mean(axis=0).sum()
        # Prediction loss of rewards.
        z_ = jnp.concatenate([z1_, z2_], axis=-1)
        rew_mean_, rew_std_ = self.model["reward"].apply(params_model["reward"], z_[:, :-1], action_, z_[:, 1:])
        loss_rew = -(gaussian_log_prob(jnp.log(rew_std_), (reward_ - rew_mean_) / rew_std_) * (1 - done_)).mean(axis=0).sum()
        return loss_kld + loss_img + loss_rew, None

    @partial(jax.jit, static_argnums=0)
    def sample_prior(
        self,
        params_model: hk.Params,
        action_: jnp.ndarray,
        key_list: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        z1_mean_ = []
        z1_std_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = self.model["z1_prior_init"].apply(params_model["z1_prior_init"], action_[:, 0])
        z1 = z1_mean + jax.random.normal(key_list[0], z1_std.shape) * z1_std
        # p(z2(0) | z1(0))
        z2_mean, z2_std = self.model["z2_init"].apply(params_model["z2_init"], z1)
        z2 = z2_mean + jax.random.normal(key_list[1], z2_std.shape) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)

        for t in range(1, action_.shape[1] + 1):
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean, z1_std = self.model["z1_prior"].apply(params_model["z1_prior"], z2, action_[:, t - 1])
            z1 = z1_mean + jax.random.normal(key_list[2 * t], z1_std.shape) * z1_std
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = self.model["z2"].apply(params_model["z2"], z1, z2, action_[:, t - 1])
            z2 = z2_mean + jax.random.normal(key_list[2 * t + 1], z2_std.shape) * z2_std

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)

        z1_mean_ = jnp.stack(z1_mean_, axis=1)
        z1_std_ = jnp.stack(z1_std_, axis=1)
        return (z1_mean_, z1_std_)

    @partial(jax.jit, static_argnums=0)
    def sample_post(
        self,
        params_model: hk.Params,
        feature_: jnp.ndarray,
        action_: jnp.ndarray,
        key_list: List[np.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z1_mean_ = []
        z1_std_ = []
        z1_ = []
        z2_ = []

        # q(z1(0) | feat(0))
        z1_mean, z1_std = self.model["z1_post_init"].apply(params_model["z1_post_init"], feature_[:, 0])
        z1 = z1_mean + jax.random.normal(key_list[0], z1_std.shape) * z1_std
        # q(z2(0) | z1(0))
        z2_mean, z2_std = self.model["z2_init"].apply(params_model["z2_init"], z1)
        z2 = z2_mean + jax.random.normal(key_list[1], z2_std.shape) * z2_std

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        for t in range(1, action_.shape[1] + 1):
            # q(z1(t+1) | feat(t+1), z2(t), a(t))
            z1_mean, z1_std = self.model["z1_post"].apply(params_model["z1_post"], feature_[:, t], z2, action_[:, t - 1])
            z1 = z1_mean + jax.random.normal(key_list[2 * t], z1_std.shape) * z1_std
            # q(z2(t+1) | z1(t+1), z2(t), a(t))
            z2_mean, z2_std = self.model["z2"].apply(params_model["z2"], z1, z2, action_[:, t - 1])
            z2 = z2_mean + jax.random.normal(key_list[2 * t + 1], z2_std.shape) * z2_std

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
        SAC.save_params(self, save_dir)
        save_params(self.params_model, os.path.join(save_dir, "params_model.npz"))

    def load_params(self, save_dir):
        SAC.load_params(self, save_dir)
        self.params_model = load_params(os.path.join(save_dir, "params_model.npz"))
