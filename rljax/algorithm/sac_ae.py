import os
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import OffPolicyActorCritic
from rljax.network import ContinuousQFunction, SACDecoder, SACEncoder, SACLinear, StateDependentGaussianPolicy
from rljax.util import (
    clip_gradient_norm,
    load_params,
    preprocess_state,
    reparameterize_gaussian_and_tanh,
    save_params,
    soft_update,
    weight_decay,
)


class SAC_AE(OffPolicyActorCritic):
    name = "SAC+AE"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=128,
        start_steps=1000,
        update_interval=1,
        tau=0.01,
        tau_ae=0.05,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_ae=1e-3,
        lr_alpha=1e-4,
        units_actor=(1024, 1024),
        units_critic=(1024, 1024),
        feature_dim=50,
        alpha_init=0.1,
        lambda_latent=1e-6,
        lambda_weight=1e-7,
        update_interval_actor=2,
        update_interval_ae=1,
        update_interval_target=2,
    ):
        assert len(state_space.shape) == 3 and state_space.shape[:2] == (84, 84)
        super(SAC_AE, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
        )

        def critic_fn(x, a):
            # Define without linear layer.
            return ContinuousQFunction(
                num_critics=2,
                hidden_units=units_critic,
            )(x, a)

        def actor_fn(x):
            # Define with linear layer.
            x = SACLinear(feature_dim=feature_dim)(x)
            return StateDependentGaussianPolicy(
                action_space=action_space,
                hidden_units=units_actor,
                log_std_min=-10.0,
                clip_log_std=False,
            )(x)

        # Encoder.
        self.encoder = hk.without_apply_rng(hk.transform(lambda s: SACEncoder(num_filters=32, num_layers=4)(s)))
        self.params_encoder = self.params_encoder_target = self.encoder.init(next(self.rng), self.fake_state)
        fake_last_conv = np.zeros((1, 32 * (43 - 2 * 4) * (43 - 2 * 4)), dtype=np.float32)

        # Linear layer for critic and decoder.
        self.linear = hk.without_apply_rng(hk.transform(lambda x: SACLinear(feature_dim=feature_dim)(x)))
        self.params_linear = self.params_linear_target = self.linear.init(next(self.rng), fake_last_conv)
        fake_feature = np.zeros((1, feature_dim), dtype=np.float32)

        # Critic.
        self.critic = hk.without_apply_rng(hk.transform(critic_fn))
        params_critic_head = self.critic.init(next(self.rng), fake_feature, self.fake_action)
        self.params_critic_head = self.params_critic_head_target = params_critic_head
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_critic)

        # Actor with linear layer.
        self.actor = hk.without_apply_rng(hk.transform(actor_fn))
        self.params_actor = self.actor.init(next(self.rng), fake_last_conv)
        opt_init, self.opt_actor = optix.adam(lr_actor)
        self.opt_state_actor = opt_init(self.params_actor)

        # Decoder.
        self.decoder = hk.without_apply_rng(hk.transform(lambda x: SACDecoder(state_space, num_filters=32, num_layers=4)(x)))
        self.params_decoder = self.decoder.init(next(self.rng), fake_feature)
        opt_init, self.opt_ae = optix.adam(lr_ae)
        self.opt_state_ae = opt_init(self.params_ae)

        # Entropy coefficient.
        self.target_entropy = -float(action_space.shape[0])
        self.log_alpha = jnp.array(np.log(alpha_init), dtype=jnp.float32)
        opt_init, self.opt_alpha = optix.adam(lr_alpha, b1=0.5)
        self.opt_state_alpha = opt_init(self.log_alpha)

        # Other parameters.
        self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
        self.lambda_latent = lambda_latent
        self.lambda_weight = lambda_weight
        self.update_interval_actor = update_interval_actor
        self.update_interval_ae = update_interval_ae
        self.update_interval_target = update_interval_target

    @property
    def params_ae(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "decoder": self.params_decoder,
        }

    @property
    def params_critic(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "critic": self.params_critic_head,
        }

    @property
    def params_critic_target(self):
        return {
            "encoder": self.params_encoder_target,
            "linear": self.params_linear_target,
            "critic": self.params_critic_head_target,
        }

    def select_action(self, state):
        action = self._select_action(self.params_encoder, self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(self.params_encoder, self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params_encoder: hk.Params,
        params_actor: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        last_conv = self.encoder.apply(params_encoder, state)
        mean, _ = self.actor.apply(params_actor, last_conv)
        return jnp.tanh(mean)

    @partial(jax.jit, static_argnums=0)
    def _explore(
        self,
        params_encoder: hk.Params,
        params_actor: hk.Params,
        key: jnp.ndarray,
        state: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        last_conv = self.encoder.apply(params_encoder, state)
        mean, log_std = self.actor.apply(params_actor, last_conv)
        return reparameterize_gaussian_and_tanh(mean, log_std, key, False)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
        self.opt_state_critic, params_critic, loss_critic, (abs_td1, _) = self._update_critic(
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
            weight1=weight,
            weight2=weight,
            key=next(self.rng),
        )
        self.params_encoder, self.params_linear, self.params_critic_head = params_critic

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td1)

        # Update actor and alpha.
        if self.learning_step % self.update_interval_actor == 0:
            self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = self._update_actor(
                opt_state_actor=self.opt_state_actor,
                params_actor=self.params_actor,
                params_critic=self.params_critic,
                log_alpha=self.log_alpha,
                state=state,
                key=next(self.rng),
            )
            self.opt_state_alpha, self.log_alpha, loss_alpha = self._update_alpha(
                opt_state_alpha=self.opt_state_alpha,
                log_alpha=self.log_alpha,
                mean_log_pi=mean_log_pi,
            )

        # Update autoencoder.
        if self.learning_step % self.update_interval_actor == 0:
            self.opt_state_ae, params_ae, loss_ae = self._update_ae(
                opt_state_ae=self.opt_state_ae,
                params_ae=self.params_ae,
                state=state,
                key=next(self.rng),
            )
            self.params_encoder, self.params_linear, self.params_decoder = params_ae

        # Update target network.
        if self.learning_step % self.update_interval_target == 0:
            self.params_encoder_target = self._update_target_ae(self.params_encoder_target, self.params_encoder)
            self.params_linear_target = self._update_target_ae(self.params_linear_target, self.params_linear)
            self.params_critic_head_target = self._update_target(self.params_critic_head_target, self.params_critic_head)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/ae", loss_ae, self.learning_step)
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
        weight1: np.ndarray,
        weight2: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_critic, (abs_td1, abs_td2)), grad_critic = jax.value_and_grad(self._loss_critic, has_aux=True)(
            params_critic,
            params_critic_target=params_critic_target,
            params_actor=params_actor,
            log_alpha=log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight1=weight1,
            weight2=weight2,
            key=key,
        )
        if self.max_grad_norm is not None:
            grad_critic = clip_gradient_norm(grad_critic, self.max_grad_norm)
        update, opt_state_critic = self.opt_critic(grad_critic, opt_state_critic)
        params_critic = optix.apply_updates(params_critic, update)
        params_critic = (params_critic["encoder"], params_critic["linear"], params_critic["critic"])
        return opt_state_critic, params_critic, loss_critic, (abs_td1, abs_td2)

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
        weight1: np.ndarray,
        weight2: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jnp.exp(log_alpha)
        # Sample next actions.
        next_last_conv = self.encoder.apply(params_critic["encoder"], next_state)
        next_mean, next_log_std = self.actor.apply(params_actor, next_last_conv)
        next_action, next_log_pi = reparameterize_gaussian_and_tanh(next_mean, next_log_std, key, True)
        # Calculate target soft q values (clipped double q) with target critic.
        next_last_conv_prime = self.encoder.apply(params_critic_target["encoder"], next_state)
        next_feature = self.linear.apply(params_critic_target["linear"], next_last_conv_prime)
        next_q1, next_q2 = self.critic.apply(params_critic_target["critic"], next_feature, next_action)
        next_q = jnp.minimum(next_q1, next_q2) - alpha * next_log_pi
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        # Calculate current soft q values with online critic.
        last_conv = self.encoder.apply(params_critic["encoder"], state)
        feature = self.linear.apply(params_critic["linear"], last_conv)
        curr_q1, curr_q2 = self.critic.apply(params_critic["critic"], feature, action)
        abs_td1 = jnp.abs(target_q - curr_q1)
        abs_td2 = jnp.abs(target_q - curr_q2)
        loss = (jnp.square(abs_td1) * weight1).mean() + (jnp.square(abs_td2) * weight2).mean()
        return loss, (jax.lax.stop_gradient(abs_td1), jax.lax.stop_gradient(abs_td2))

    @partial(jax.jit, static_argnums=0)
    def _update_actor(
        self,
        opt_state_actor: Any,
        params_actor: hk.Params,
        params_critic: hk.Params,
        log_alpha: jnp.ndarray,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_actor, mean_log_pi), grad_actor = jax.value_and_grad(self._loss_actor, has_aux=True)(
            params_actor,
            params_critic=params_critic,
            log_alpha=log_alpha,
            state=state,
            key=key,
        )
        if self.max_grad_norm is not None:
            grad_actor = clip_gradient_norm(grad_actor, self.max_grad_norm)
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
        key: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        alpha = jax.lax.stop_gradient(jnp.exp(log_alpha))
        last_conv = jax.lax.stop_gradient(self.encoder.apply(params_critic["encoder"], state))
        # Sample actions.
        mean, log_std = self.actor.apply(params_actor, last_conv)
        action, log_pi = reparameterize_gaussian_and_tanh(mean, log_std, key, True)
        # Calculate soft q values with online critic.
        feature = self.linear.apply(params_critic["linear"], last_conv)
        q1, q2 = self.critic.apply(params_critic["critic"], feature, action)
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
    def _update_ae(
        self,
        opt_state_ae: Any,
        params_ae: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray]:
        loss_ae, grad_ae = jax.value_and_grad(self._loss_ae)(
            params_ae,
            state=state,
            key=key,
        )
        if self.max_grad_norm is not None:
            grad_ae = clip_gradient_norm(grad_ae, self.max_grad_norm)
        update, opt_state_ae = self.opt_ae(grad_ae, opt_state_ae)
        params_ae = optix.apply_updates(params_ae, update)
        params_ae = (params_ae["encoder"], params_ae["linear"], params_ae["decoder"])
        return opt_state_ae, params_ae, loss_ae

    @partial(jax.jit, static_argnums=0)
    def _loss_ae(
        self,
        params_ae: hk.Params,
        state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Preprocess states.
        target = preprocess_state(state, key)
        # Reconstruct states.
        last_conv = self.encoder.apply(params_ae["encoder"], state)
        feature = self.linear.apply(params_ae["linear"], last_conv)
        reconst = self.decoder.apply(params_ae["decoder"], feature)

        # MSE for reconstruction errors.
        loss_reconst = jnp.square(target - reconst).mean()
        # L2 penalty of latent representations following RAE.
        loss_latent = 0.5 * jnp.square(feature).sum(axis=1).mean()
        # Weight decay for the decoder.
        loss_weight = weight_decay(params_ae["decoder"])
        # RAE loss is reconstruction loss plus the reglarizations.
        # (i.e. L2 penalty of latent representations + weight decay.)
        return loss_reconst + self.lambda_latent * loss_latent + self.lambda_weight * loss_weight

    def save_params(self, save_dir):
        save_params(self.params_encoder, os.path.join(save_dir, "params_encoder.npz"))
        save_params(self.params_linear, os.path.join(save_dir, "params_linear.npz"))
        save_params(self.params_decoder, os.path.join(save_dir, "params_decoder.npz"))
        save_params(self.params_critic_head, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_encoder = self.params_encoder_target = load_params(os.path.join(save_dir, "params_encoder.npz"))
        self.params_linear = self.params_linear_target = load_params(os.path.join(save_dir, "params_linear.npz"))
        self.params_decoder = load_params(os.path.join(save_dir, "params_decoder.npz"))
        self.params_critic_head = self.params_critic_head_target = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = load_params(os.path.join(save_dir, "params_actor.npz"))
