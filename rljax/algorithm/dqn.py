from functools import partial
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import optix

from rljax.algorithm.base import QLearning
from rljax.network import DiscreteQFunction
from rljax.util import get_q_at_action, huber, optimize


class DQN(QLearning):
    name = "DQN"

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
        batch_size=32,
        start_steps=50000,
        update_interval=4,
        update_interval_target=8000,
        eps=0.01,
        eps_eval=0.001,
        eps_decay_steps=250000,
        fn=None,
        lr=2.5e-4,
        units=(512,),
        loss_type="huber",
        dueling_net=False,
        double_q=False,
    ):
        assert loss_type in ["l2", "huber"]
        super(DQN, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            batch_size=batch_size,
            use_per=use_per,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
            eps=eps,
            eps_eval=eps_eval,
            eps_decay_steps=eps_decay_steps,
        )

        if fn is None:

            def fn(s):
                return DiscreteQFunction(
                    action_space=action_space,
                    num_critics=1,
                    hidden_units=units,
                    dueling_net=dueling_net,
                )(s)

        self.net = hk.without_apply_rng(hk.transform(fn))
        self.params = self.params_target = self.net.init(next(self.rng), *self.fake_args)
        opt_init, self.opt = optix.adam(lr, eps=0.01 / batch_size)
        self.opt_state = opt_init(self.params)

        # Other parameters.
        self.loss_type = loss_type
        self.double_q = double_q
        if not hasattr(self, "random_update"):
            # DQN._loss() doesn't need a random key.
            self.random_update = False

    @partial(jax.jit, static_argnums=0)
    def _forward(
        self,
        params: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        q_s = self.net.apply(params, state)
        return jnp.argmax(q_s, axis=1)

    def update(self, writer=None):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        kwargs = {"key1": next(self.rng), "key2": next(self.rng)} if self.random_update else {}
        self.opt_state, self.params, loss, abs_td = optimize(
            self._loss,
            self.opt,
            self.opt_state,
            self.params,
            self.max_grad_norm,
            params_target=self.params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
            **kwargs,
        )

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(abs_td)

        # Update target network.
        if self.agent_step % self.update_interval_target == 0:
            self.params_target = self._update_target(self.params_target, self.params)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/q", loss, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _loss(
        self,
        params: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.double_q:
            # Calculate greedy actions with online network.
            next_action = self._forward(params, next_state)[..., None]
            # Then calculate max q values with target network.
            next_q = get_q_at_action(self.net.apply(params_target, next_state), next_action)
        else:
            # Calculate greedy actions and max q values with target network.
            next_q = jnp.max(self.net.apply(params_target, next_state), axis=1, keepdims=True)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q = get_q_at_action(self.net.apply(params, state), action)

        td = target_q - curr_q
        if self.loss_type == "l2":
            loss = jnp.mean(jnp.square(td) * weight)
        elif self.loss_type == "huber":
            loss = jnp.mean(huber(td) * weight)
        return loss, jax.lax.stop_gradient(jnp.abs(td))
