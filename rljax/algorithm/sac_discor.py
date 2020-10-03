import os
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQFunction
from rljax.util import load_params, save_params


class SAC_DisCor(SAC):
    name = "SAC+DisCor"

    def __init__(
        self,
        num_steps,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        lr_error=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_error=(256, 256, 256),
        error_init=10.0,
    ):
        assert nstep == 1
        super(SAC_DisCor, self).__init__(
            num_steps=num_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=False,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            units_actor=units_actor,
            units_critic=units_critic,
        )

        def error_fn(s, a):
            return ContinuousQFunction(
                num_critics=2,
                hidden_units=units_error,
            )(s, a)

        # Error model.
        self.error = hk.without_apply_rng(hk.transform(error_fn))
        opt_init, self.opt_error = optix.adam(lr_error)
        self.params_error = self.params_error_target = self.error.init(next(self.rng), self.fake_state, self.fake_action)
        self.opt_state_error = opt_init(self.params_error)

        # Running mean of errors.
        self.rm_error1 = jnp.array(error_init, dtype=jnp.float32)
        self.rm_error2 = jnp.array(error_init, dtype=jnp.float32)

    def update(self, writer=None):
        self.learning_step += 1
        _, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Calculate weights.
        weight1, weight2 = self.calculate_weight(
            params_actor=self.params_actor,
            params_error_target=self.params_error_target,
            rm_error1=self.rm_error1,
            rm_error2=self.rm_error2,
            done=done,
            next_state=next_state,
            rng=next(self.rng),
        )

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, (abs_td1, abs_td2) = self._update_critic(
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
            weight1=weight1,
            weight2=weight2,
            rng=next(self.rng),
        )

        # Update error model.
        self.opt_state_error, self.params_error, loss_error, (mean_error1, mean_error2) = self._update_error(
            opt_state_error=self.opt_state_error,
            params_error=self.params_error,
            params_error_target=self.params_error_target,
            params_actor=self.params_actor,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td1=abs_td1,
            abs_td2=abs_td2,
            rng=next(self.rng),
        )

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

        # Update target networks.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)
        self.params_error_target = self._update_target(self.params_error_target, self.params_error)
        self.rm_error1 = self._update_target(self.rm_error1, mean_error1)
        self.rm_error2 = self._update_target(self.rm_error2, mean_error2)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("loss/error", loss_error, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)
            writer.add_scalar("stat/rm_error1", self.rm_error1, self.learning_step)
            writer.add_scalar("stat/rm_error2", self.rm_error2, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def sample_next_error(
        self,
        params_actor: hk.Params,
        params_error_target: hk.Params,
        next_state: np.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate actions.
        next_action = self._explore(params_actor, rng, next_state)
        # Calculate errors.
        return self.error.apply(params_error_target, next_state, next_action)

    @partial(jax.jit, static_argnums=0)
    def calculate_weight(
        self,
        params_actor: hk.Params,
        params_error_target: hk.Params,
        rm_error1: jnp.ndarray,
        rm_error2: jnp.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Calculate next errors.
        next_error1, next_error2 = self.sample_next_error(params_actor, params_error_target, next_state, rng)
        # Terms inside the exponent of importance weights.
        x1 = -(1.0 - done) * self.gamma * next_error1 / rm_error1
        x2 = -(1.0 - done) * self.gamma * next_error2 / rm_error2
        # Calculate importance weights.
        weight1 = jax.lax.stop_gradient(jax.nn.softmax(x1, axis=0) * x1.shape[0])
        weight2 = jax.lax.stop_gradient(jax.nn.softmax(x2, axis=0) * x2.shape[0])
        return weight1, weight2

    @partial(jax.jit, static_argnums=0)
    def _update_error(
        self,
        opt_state_error: Any,
        params_error: hk.Params,
        params_error_target: hk.Params,
        params_actor: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        abs_td1: jnp.ndarray,
        abs_td2: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_error, (mean_error1, mean_error2)), grad_error = jax.value_and_grad(self._loss_error, has_aux=True)(
            params_error,
            params_error_target=params_error_target,
            params_actor=params_actor,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td1=abs_td1,
            abs_td2=abs_td2,
            rng=rng,
        )
        update, opt_state_error = self.opt_error(grad_error, opt_state_error)
        params_error = optix.apply_updates(params_error, update)
        return opt_state_error, params_error, loss_error, (mean_error1, mean_error2)

    @partial(jax.jit, static_argnums=0)
    def _loss_error(
        self,
        params_error: hk.Params,
        params_error_target: hk.Params,
        params_actor: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        abs_td1: np.ndarray,
        abs_td2: np.ndarray,
        rng: jnp.ndarray,
    ) -> jnp.ndarray:
        # Calculate next errors.
        next_error1, next_error2 = self.sample_next_error(params_actor, params_error_target, next_state, rng)
        # Calculate target errors.
        target_error1 = jax.lax.stop_gradient(abs_td1 + (1.0 - done) * self.gamma * next_error1)
        target_error2 = jax.lax.stop_gradient(abs_td2 + (1.0 - done) * self.gamma * next_error2)
        # Calculate current errors.
        curr_error1, curr_error2 = self.error.apply(params_error, state, action)
        loss = jnp.square(curr_error1 - target_error1).mean() + jnp.square(curr_error2 - target_error2).mean()
        mean_error1, mean_error2 = jax.lax.stop_gradient(curr_error1.mean()), jax.lax.stop_gradient(curr_error2.mean())
        return loss, (mean_error1, mean_error2)

    def save_params(self, save_dir):
        super(SAC_DisCor, self).save_params(save_dir)
        save_params(self.params_error, os.path.join(save_dir, "params_error.npz"))

    def load_params(self, save_dir):
        super(SAC_DisCor, self).load_params(save_dir)
        self.params_error = self.params_error_target = load_params(os.path.join(save_dir, "params_error.npz"))
