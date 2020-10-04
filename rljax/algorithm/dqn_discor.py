import os
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.dqn import DQN
from rljax.network import DiscreteQFunction
from rljax.util import get_q_at_action, load_params, save_params, soft_update


class DQN_DisCor(DQN):
    name = "DQN+DisCor"

    def __init__(
        self,
        num_steps,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        batch_size=32,
        start_steps=50000,
        update_interval=4,
        update_interval_target=8000,
        eps=0.01,
        eps_eval=0.001,
        eps_decay_steps=250000,
        lr=2.5e-4,
        lr_error=2.5e-4,
        units=(512,),
        units_error=(512, 512),
        loss_type="l2",
        dueling_net=False,
        double_q=False,
        tau=5e-3,
        error_init=10.0,
    ):
        assert nstep == 1
        super(DQN_DisCor, self).__init__(
            num_steps=num_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            batch_size=batch_size,
            use_per=False,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
            eps=eps,
            eps_eval=eps_eval,
            eps_decay_steps=eps_decay_steps,
            lr=lr,
            units=units,
            loss_type=loss_type,
            dueling_net=dueling_net,
            double_q=double_q,
        )

        def error_fn(s):
            return DiscreteQFunction(
                action_space=action_space,
                num_critics=1,
                hidden_units=units_error,
            )(s)

        # Error model.
        self.error = hk.without_apply_rng(hk.transform(error_fn))
        self.params_error = self.params_error_target = self.error.init(next(self.rng), self.fake_state)
        opt_init, self.opt_error = optix.adam(lr_error)
        self.opt_state_error = opt_init(self.params_error)

        # Running mean of errors.
        self.rm_error = jnp.array(error_init, dtype=jnp.float32)
        self._update_target_error = jax.jit(partial(soft_update, tau=tau))

    def update(self, writer=None):
        self.learning_step += 1
        _, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch
        weight = self.calculate_weight(
            params_target=self.params_target,
            params_error_target=self.params_error_target,
            rm_error=self.rm_error,
            done=done,
            next_state=next_state,
        )

        self.opt_state, self.params, loss, abs_td = self._update(
            opt_state=self.opt_state,
            params=self.params,
            params_target=self.params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )

        self.opt_state_error, self.params_error, loss_error, mean_error = self._update_error(
            opt_state_error=self.opt_state_error,
            params_error=self.params_error,
            params_error_target=self.params_error_target,
            params_target=self.params_target,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td=abs_td,
        )

        # Update target network.
        self.rm_error = self._update_target_error(self.rm_error, mean_error)
        if self.env_step % self.update_interval_target == 0:
            self.params_target = self._update_target(self.params_target, self.params)
            self.params_error_target = self._update_target(self.params_error_target, self.params_error)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/q", loss, self.learning_step)
            writer.add_scalar("loss/error", loss_error, self.learning_step)
            writer.add_scalar("stat/rm_error", self.rm_error, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def sample_next_error(
        self,
        params_target: hk.Params,
        params_error_target: hk.Params,
        next_state: np.ndarray,
    ) -> jnp.ndarray:
        # Calculate actions.
        next_action = self._forward(params_target, next_state)
        # Calculate errors.
        return get_q_at_action(self.error.apply(params_error_target, next_state), next_action)

    @partial(jax.jit, static_argnums=0)
    def calculate_weight(
        self,
        params_target: hk.Params,
        params_error_target: hk.Params,
        rm_error: jnp.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
    ) -> jnp.ndarray:
        # Calculate next errors.
        next_error = self.sample_next_error(params_target, params_error_target, next_state)
        # Terms inside the exponent of importance weights.
        x = -(1.0 - done) * self.gamma * next_error / rm_error
        # Calculate importance weights.
        return jax.lax.stop_gradient(jax.nn.softmax(x, axis=0) * x.shape[0])

    @partial(jax.jit, static_argnums=0)
    def _update_error(
        self,
        opt_state_error: Any,
        params_error: hk.Params,
        params_error_target: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        abs_td: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss_error, mean_error), grad_error = jax.value_and_grad(self._loss_error, has_aux=True)(
            params_error,
            params_error_target=params_error_target,
            params_target=params_target,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td=abs_td,
        )
        update, opt_state_error = self.opt_error(grad_error, opt_state_error)
        params_error = optix.apply_updates(params_error, update)
        return opt_state_error, params_error, loss_error, mean_error

    @partial(jax.jit, static_argnums=0)
    def _loss_error(
        self,
        params_error: hk.Params,
        params_error_target: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        abs_td: np.ndarray,
    ) -> jnp.ndarray:
        # Calculate next errors.
        next_error = self.sample_next_error(params_target, params_error_target, next_state)
        # Calculate target errors.
        target_error = jax.lax.stop_gradient(abs_td + (1.0 - done) * self.gamma * next_error)
        # Calculate current errors.
        curr_error = get_q_at_action(self.error.apply(params_error, state), action)
        loss = jnp.square(curr_error - target_error).mean()
        mean_error = jax.lax.stop_gradient(curr_error.mean())
        return loss, mean_error

    def save_params(self, save_dir):
        super(DQN_DisCor, self).save_params(save_dir)
        save_params(self.params_error, os.path.join(save_dir, "params_error.npz"))

    def load_params(self, save_dir):
        super(DQN_DisCor, self).load_params(save_dir)
        self.params_error = self.params_error_target = load_params(os.path.join(save_dir, "params_error.npz"))
