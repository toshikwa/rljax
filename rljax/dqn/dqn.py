from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from flax import nn, optim
from rljax.common.base_class import DiscreteOffPolicyAlgorithm
from rljax.common.utils import soft_update, update_network
from rljax.dqn.network import build_dqn


@jax.jit
def _calculate_double_q(
    q: jnp.ndarray,
    q_target: jnp.ndarray,
) -> jnp.ndarray:
    action = jnp.argmax(q)
    return q_target[[action]]


def grad_fn(
    dqn: nn.Model,
    dqn_target: nn.Model,
    weight: jnp.ndarray,
    discount: float,
    double_q: bool,
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    next_state: jnp.ndarray,
) -> nn.Model:
    if double_q:
        next_q = jax.vmap(_calculate_double_q)(dqn(next_state), dqn_target(next_state))
    else:
        next_q = jnp.max(dqn_target(next_state), axis=1, keepdims=True)
    target_q = jax.lax.stop_gradient(reward + (1.0 - done) * discount * next_q)

    def _loss(action, curr_q, target_q):
        return jnp.abs(target_q - curr_q[action])

    def loss_fn(dqn):
        curr_q = dqn(state)
        td_error = jax.vmap(_loss)(action, curr_q, target_q)
        return jnp.mean(jnp.square(td_error) * weight), jax.lax.stop_gradient(td_error)

    grad, td_error = jax.grad(loss_fn, has_aux=True)(dqn)
    return grad, td_error


class DQN(DiscreteOffPolicyAlgorithm):
    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma=0.99,
        nstep=1,
        buffer_size=10 ** 6,
        use_per=False,
        batch_size=256,
        start_steps=1000,
        eps=0.01,
        eps_eval=0.001,
        update_interval=1,
        update_interval_target=1000,
        lr=1e-4,
        units=(512,),
        dueling_net=True,
        double_q=True,
    ):
        super(DQN, self).__init__(
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
        )

        # DQN.
        rng = next(self.rng)
        dqn = build_dqn(
            state_dim=state_space.shape[0],
            action_dim=action_space.n,
            rng_init=rng,
            hidden_units=units,
            dueling_net=dueling_net,
        )
        self.optim = jax.device_put(optim.Adam(learning_rate=lr).create(dqn))

        # Target network.
        self.dqn_target = jax.device_put(
            build_dqn(
                state_dim=state_space.shape[0],
                action_dim=action_space.n,
                rng_init=rng,
                hidden_units=units,
                dueling_net=dueling_net,
            )
        )

        # Compile function.
        self.grad_fn = jax.jit(partial(grad_fn, discount=self.discount, double_q=double_q))

        # Other parameters.
        self.eps = eps
        self.eps_eval = eps_eval

    def select_action(self, state):
        if np.random.rand() < self.eps_eval:
            action = self.action_space.sample()
        else:
            state = jax.device_put(state[None, ...])
            action = self.dqn(state)
            action = np.argmax(action[0])
        return action

    def step(self, env, state, t, step):
        t += 1

        if np.random.rand() < self.eps:
            action = env.action_space.sample()
        else:
            action = self.select_action(state)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, next_state, done)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        weight, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Update.
        grad, td_error = self.grad_fn(
            dqn=self.dqn,
            dqn_target=self.dqn_target,
            weight=weight,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )
        self.optim = update_network(self.optim, grad)

        # Update priority.
        if self.use_per:
            self.buffer.update_priority(td_error)

        # Update target network.
        if (self.learning_steps * self.update_interval) % self.update_interval_target == 0:
            self.dqn_target = soft_update(self.dqn_target, self.dqn, 1.0)

    @property
    def dqn(self):
        return self.optim.target

    def __str__(self):
        return "dqn" if not self.use_per else "dqn_per"
