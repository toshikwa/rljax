from functools import partial
from typing import Any, Tuple

import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
from jax import nn
from jax.experimental import optix
from rljax.algorithm.base import QLearning
from rljax.network.critic import DiscreteQFunction, DQNBody
from rljax.utils import get_q_at_action


def build_dqn(state_space, action_space, hidden_units, dueling_net):
    def _func(x):
        if len(state_space.shape) == 3:
            x = DQNBody()(x)
        return DiscreteQFunction(
            action_dim=action_space.n,
            num_critics=1,
            hidden_units=hidden_units,
            hidden_activation=nn.relu,
            dueling_net=dueling_net,
        )(x)

    fake_input = state_space.sample()
    if len(state_space.shape) == 1:
        fake_input = fake_input.astype(np.float32)
    return hk.without_apply_rng(hk.transform(_func)), fake_input[None, ...]


class DQN(QLearning):
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
        batch_size=32,
        start_steps=50000,
        update_interval=4,
        update_interval_target=10000,
        eps=0.01,
        eps_eval=0.001,
        lr=2.5e-4,
        units=(512,),
        dueling_net=True,
        double_q=True,
    ):
        assert update_interval_target % update_interval == 0
        super(DQN, self).__init__(
            num_steps=num_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
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
        )

        # DQN.
        self.q_net, fake_input = build_dqn(state_space, action_space, units, dueling_net)
        opt_init, self.opt = optix.adam(lr)
        self.params = self.params_target = self.q_net.init(next(self.rng), fake_input)
        self.opt_state = opt_init(self.params)

        # Other parameters.
        self.double_q = double_q

    @partial(jax.jit, static_argnums=0)
    def _select_action(
        self,
        params: hk.Params,
        state: np.ndarray,
    ) -> jnp.ndarray:
        s_q = self.q_net.apply(params, state)
        return jnp.argmax(s_q, axis=1)

    def update(self, writer):
        self.learning_step += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        self.opt_state, self.params, loss, error = self._update(
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

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(error)

        # Update target network.
        if self.env_step % self.update_interval_target == 0:
            self.params_target = self._update_target(self.params_target, self.params)

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/q", loss, self.learning_step)

    @partial(jax.jit, static_argnums=0)
    def _update(
        self,
        opt_state: Any,
        params: hk.Params,
        params_target: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_state: np.ndarray,
        weight: np.ndarray,
    ) -> Tuple[Any, hk.Params, jnp.ndarray, jnp.ndarray]:
        (loss, error), grad = jax.value_and_grad(self._loss, has_aux=True)(
            params,
            params_target=params_target,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight,
        )
        update, opt_state = self.opt(grad, opt_state)
        params = optix.apply_updates(params, update)
        return opt_state, params, loss, error

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
            # calculate greedy actions with online network.
            next_action = jnp.argmax(self.q_net.apply(params, next_state), axis=1)[..., None]
            # Then calculate max q values with target network.
            next_q = get_q_at_action(self.q_net.apply(params_target, next_state), next_action)
        else:
            # calculate greedy actions and max q values with target network.
            next_q = jnp.max(self.q_net.apply(params_target, next_state), axis=1, keepdims=True)
        target_q = jax.lax.stop_gradient(reward + (1.0 - done) * self.discount * next_q)
        curr_q = get_q_at_action(self.q_net.apply(params, state), action)
        error = jnp.abs(target_q - curr_q)
        return jnp.mean(jnp.square(error) * weight), jax.lax.stop_gradient(error)

    def __str__(self):
        return "DQN" if not self.use_per else "DQN+PER"
