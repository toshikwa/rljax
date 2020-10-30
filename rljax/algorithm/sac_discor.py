import os
from functools import partial
from typing import List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.sac import SAC
from rljax.network import ContinuousQFunction
from rljax.util import load_params, optimize, save_params


class SAC_DisCor(SAC):
    name = "SAC+DisCor"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        num_critics=2,
        buffer_size=10 ** 6,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        fn_error=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        lr_error=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_error=(256, 256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        d2rl=False,
        init_alpha=1.0,
        init_error=10.0,
        adam_b1_alpha=0.9,
    ):
        super(SAC_DisCor, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=1,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=False,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            units_actor=units_actor,
            units_critic=units_critic,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            d2rl=d2rl,
            init_alpha=init_alpha,
            adam_b1_alpha=adam_b1_alpha,
        )

        if fn_error is None:

            def fn_error(s, a):
                return ContinuousQFunction(
                    num_critics=num_critics,
                    hidden_units=units_error,
                )(s, a)

        # Error model.
        self.error = hk.without_apply_rng(hk.transform(fn_error))
        self.params_error = self.params_error_target = self.error.init(next(self.rng), *self.fake_args_critic)
        opt_init, self.opt_error = optix.adam(lr_error)
        self.opt_state_error = opt_init(self.params_error)
        # Running mean of errors.
        self.rm_error_list = [jnp.array(init_error, dtype=jnp.float32) for _ in range(num_critics)]

    def update(self, writer=None):
        self.learning_step += 1
        _, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Calculate weights.
        weight = self.calculate_weight(
            params_actor=self.params_actor,
            params_error_target=self.params_error_target,
            rm_error_list=self.rm_error_list,
            done=done,
            next_state=next_state,
            key=next(self.rng),
        )

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, abs_td_list = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_critic_target=self.params_critic_target,
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

        # Update error model.
        self.opt_state_error, self.params_error, loss_error, mean_error_list = optimize(
            self._loss_error,
            self.opt_error,
            self.opt_state_error,
            self.params_error,
            self.max_grad_norm,
            params_error_target=self.params_error_target,
            params_actor=self.params_actor,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td_list=abs_td_list,
            key=next(self.rng),
        )

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
            self._loss_actor,
            self.opt_actor,
            self.opt_state_actor,
            self.params_actor,
            self.max_grad_norm,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            state=state,
            **self.kwargs_actor,
        )

        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            None,
            mean_log_pi=mean_log_pi,
        )

        # Update target networks.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)
        self.params_error_target = self._update_target(self.params_error_target, self.params_error)
        self.rm_error_list = self._update_target(self.rm_error_list, mean_error_list)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("loss/error", loss_error, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)
            for i, rm_error in enumerate(self.rm_error_list):
                writer.add_scalar(f"stat/rm_error{i+1}", rm_error, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        q_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight: np.ndarray,
    ) -> jnp.ndarray:
        loss_critic = 0.0
        for i, q in enumerate(q_list):
            loss_critic += (jnp.square(target - q) * weight[i]).mean()
        abs_td = jax.lax.stop_gradient(jnp.abs(target - q_list[0]))
        return loss_critic, abs_td

    @partial(jax.jit, static_argnums=0)
    def sample_next_error(
        self,
        params_actor: hk.Params,
        params_error_target: hk.Params,
        next_state: np.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        next_action = self._explore(params_actor, key, next_state)
        return self.error.apply(params_error_target, next_state, next_action)

    @partial(jax.jit, static_argnums=0)
    def calculate_weight(
        self,
        params_actor: hk.Params,
        params_error_target: hk.Params,
        rm_error_list: List[jnp.ndarray],
        done: np.ndarray,
        next_state: np.ndarray,
        key: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        next_error_list = self.sample_next_error(params_actor, params_error_target, next_state, key)
        weight = []
        for next_error, rm_error in zip(next_error_list, rm_error_list):
            x = -(1.0 - done) * self.gamma * next_error / rm_error
            weight.append(jax.lax.stop_gradient(jax.nn.softmax(x, axis=0)) * x.shape[0])
        return weight

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
        abs_td_list: List[np.ndarray],
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        curr_error_list = self.error.apply(params_error, state, action)
        next_error_list = self.sample_next_error(params_actor, params_error_target, next_state, key)
        loss = 0.0
        mean_error_list = []
        for curr_error, next_error, abs_td in zip(curr_error_list, next_error_list, abs_td_list):
            target_error = jax.lax.stop_gradient(abs_td + (1.0 - done) * self.gamma * next_error)
            loss += jnp.square(curr_error - target_error).mean()
            mean_error_list.append(curr_error.mean())
        return loss, mean_error_list

    def save_params(self, save_dir):
        super(SAC_DisCor, self).save_params(save_dir)
        save_params(self.params_error, os.path.join(save_dir, "params_error.npz"))

    def load_params(self, save_dir):
        super(SAC_DisCor, self).load_params(save_dir)
        self.params_error = self.params_error_target = load_params(os.path.join(save_dir, "params_error.npz"))
