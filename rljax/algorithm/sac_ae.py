import os
from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQFunction, SACDecoder, SACEncoder, SACLinear, StateDependentGaussianPolicy
from rljax.util import fake_action, fake_state, load_params, optimize, preprocess_state, save_params, soft_update, weight_decay


class SAC_AE(SAC):
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
        fn_actor=None,
        fn_critic=None,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_ae=1e-3,
        lr_alpha=1e-4,
        units_actor=(1024, 1024),
        units_critic=(1024, 1024),
        d2rl=False,
        init_alpha=0.1,
        adam_b1_alpha=0.5,
        feature_dim=50,
        lambda_latent=1e-6,
        lambda_weight=1e-7,
        update_interval_actor=2,
        update_interval_ae=1,
        update_interval_target=2,
    ):
        assert len(state_space.shape) == 3 and state_space.shape[:2] == (84, 84)
        assert (state_space.high == 255).all()
        if d2rl:
            self.name += "-D2RL"

        if fn_critic is None:

            def fn_critic(x, a):
                # Define without linear layer.
                return ContinuousQFunction(
                    num_critics=2,
                    hidden_units=units_critic,
                    d2rl=d2rl,
                )(x, a)

        if fn_actor is None:

            def fn_actor(x):
                # Define with linear layer.
                x = SACLinear(feature_dim=feature_dim)(x)
                return StateDependentGaussianPolicy(
                    action_space=action_space,
                    hidden_units=units_actor,
                    log_std_min=-10.0,
                    clip_log_std=False,
                    d2rl=d2rl,
                )(x)

        fake_feature = jnp.empty((1, feature_dim))
        fake_last_conv = jnp.empty((1, 39200))
        if not hasattr(self, "fake_args_critic"):
            self.fake_args_critic = (fake_feature, fake_action(action_space))
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_last_conv,)

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
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            init_alpha=init_alpha,
            adam_b1_alpha=adam_b1_alpha,
        )
        # Encoder.
        self.encoder = hk.without_apply_rng(hk.transform(lambda s: SACEncoder(num_filters=32, num_layers=4)(s)))
        self.params_encoder = self.params_encoder_target = self.encoder.init(next(self.rng), fake_state(state_space))

        # Linear layer for critic and decoder.
        self.linear = hk.without_apply_rng(hk.transform(lambda x: SACLinear(feature_dim=feature_dim)(x)))
        self.params_linear = self.params_linear_target = self.linear.init(next(self.rng), fake_last_conv)

        # Decoder.
        self.decoder = hk.without_apply_rng(hk.transform(lambda x: SACDecoder(state_space, num_filters=32, num_layers=4)(x)))
        self.params_decoder = self.decoder.init(next(self.rng), fake_feature)
        opt_init, self.opt_ae = optix.adam(lr_ae)
        self.opt_state_ae = opt_init(self.params_ae)

        # Re-define the optimizer for critic.
        opt_init, self.opt_critic = optix.adam(lr_critic)
        self.opt_state_critic = opt_init(self.params_entire_critic)

        # Other parameters.
        self._update_target_ae = jax.jit(partial(soft_update, tau=tau_ae))
        self.lambda_latent = lambda_latent
        self.lambda_weight = lambda_weight
        self.update_interval_actor = update_interval_actor
        self.update_interval_ae = update_interval_ae
        self.update_interval_target = update_interval_target

    def select_action(self, state):
        last_conv = self._preprocess(self.params_encoder, state[None, ...])
        action = self._select_action(self.params_actor, last_conv)
        return np.array(action[0])

    def explore(self, state):
        last_conv = self._preprocess(self.params_encoder, state[None, ...])
        action = self._explore(self.params_actor, next(self.rng), last_conv)
        return np.array(action[0])

    @partial(jax.jit, static_argnums=0)
    def _preprocess(
        self,
        params_encoder: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        return self.encoder.apply(params_encoder, state)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update critic.
        self.opt_state_critic, params_entire_critic, loss_critic, abs_td = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_entire_critic,
            self.max_grad_norm,
            params_critic_target=self.params_entire_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **self.kwargs_critic,
        )
        self.params_encoder = params_entire_critic["encoder"]
        self.params_linear = params_entire_critic["linear"]
        self.params_critic = params_entire_critic["critic"]

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update actor and alpha.
        if self.learning_step % self.update_interval_actor == 0:
            self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
                self._loss_actor,
                self.opt_actor,
                self.opt_state_actor,
                self.params_actor,
                self.max_grad_norm,
                params_critic=self.params_entire_critic,
                log_alpha=self.log_alpha,
                state=state,
                **self.kwargs_actor,
            )
            self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
                self._loss_alpha,
                self.opt_alpha,
                self.opt_state_alpha,
                self.log_alpha,
                None,
                mean_log_pi=mean_log_pi,
            )

        # Update autoencoder.
        if self.learning_step % self.update_interval_actor == 0:
            self.opt_state_ae, params_ae, loss_ae, _ = optimize(
                self._loss_ae,
                self.opt_ae,
                self.opt_state_ae,
                self.params_ae,
                self.max_grad_norm,
                state=state,
                key=next(self.rng),
            )
            self.params_encoder = params_ae["encoder"]
            self.params_linear = params_ae["linear"]
            self.params_decoder = params_ae["decoder"]

        # Update target network.
        if self.learning_step % self.update_interval_target == 0:
            self.params_encoder_target = self._update_target_ae(self.params_encoder_target, self.params_encoder)
            self.params_linear_target = self._update_target_ae(self.params_linear_target, self.params_linear)
            self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/ae", loss_ae, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _calculate_target(
        self,
        params_critic_target: hk.Params,
        log_alpha: jnp.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        next_action: jnp.ndarray,
        next_log_pi: jnp.ndarray,
    ) -> jnp.ndarray:
        alpha = jnp.exp(log_alpha)
        next_last_conv_prime = self.encoder.apply(params_critic_target["encoder"], next_state)
        next_feature = self.linear.apply(params_critic_target["linear"], next_last_conv_prime)
        next_q_list = self.critic.apply(params_critic_target["critic"], next_feature, next_action)
        next_q = jnp.asarray(next_q_list).min(axis=0) - alpha * next_log_pi
        return jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)

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
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_last_conv = self.encoder.apply(params_critic["encoder"], next_state)
        next_action, next_log_pi = self._sample_action(params_actor, key, next_last_conv)
        target = self._calculate_target(params_critic_target, log_alpha, reward, done, next_state, next_action, next_log_pi)
        last_conv = self.encoder.apply(params_critic["encoder"], state)
        feature = self.linear.apply(params_critic["linear"], last_conv)
        curr_q_list = self.critic.apply(params_critic["critic"], feature, action)
        loss = 0.0
        for curr_q in curr_q_list:
            loss += (jnp.square(target - curr_q) * weight).mean()
        abs_td = jax.lax.stop_gradient(jnp.abs(target - curr_q[0]))
        return loss, abs_td

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
        action, log_pi = self._sample_action(params_actor, key, last_conv)
        feature = self.linear.apply(params_critic["linear"], last_conv)
        mean_q = jnp.asarray(self.critic.apply(params_critic["critic"], feature, action)).min(axis=0).mean()
        mean_log_pi = log_pi.mean()
        return alpha * mean_log_pi - mean_q, jax.lax.stop_gradient(mean_log_pi)

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
        return loss_reconst + self.lambda_latent * loss_latent + self.lambda_weight * loss_weight, None

    @property
    def params_ae(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "decoder": self.params_decoder,
        }

    @property
    def params_entire_critic(self):
        return {
            "encoder": self.params_encoder,
            "linear": self.params_linear,
            "critic": self.params_critic,
        }

    @property
    def params_entire_critic_target(self):
        return {
            "encoder": self.params_encoder_target,
            "linear": self.params_linear_target,
            "critic": self.params_critic_target,
        }

    def save_params(self, save_dir):
        super().save_params(save_dir)
        save_params(self.params_encoder, os.path.join(save_dir, "params_encoder.npz"))
        save_params(self.params_linear, os.path.join(save_dir, "params_linear.npz"))
        save_params(self.params_decoder, os.path.join(save_dir, "params_decoder.npz"))

    def load_params(self, save_dir):
        super().load_params(save_dir)
        self.params_encoder = self.params_encoder_target = load_params(os.path.join(save_dir, "params_encoder.npz"))
        self.params_linear = self.params_linear_target = load_params(os.path.join(save_dir, "params_linear.npz"))
        self.params_decoder = load_params(os.path.join(save_dir, "params_decoder.npz"))
