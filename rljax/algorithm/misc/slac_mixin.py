import os
from abc import abstractmethod
from collections import deque

import jax.numpy as jnp
import numpy as np

from rljax.buffer import SLACReplayBuffer
from rljax.util import fake_action, load_params, save_params


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_space, action_space, num_sequences):
        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.num_sequences = num_sequences
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)

    def reset_episode(self, state):
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


class SlacMixIn:
    """
    MixIn for SLAC-based algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        num_sequences,
        buffer_size,
        batch_size_sac,
        batch_size_model,
        initial_learning_steps,
        feature_dim,
        z1_dim,
        z2_dim,
    ):
        self.buffer = SLACReplayBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            num_sequences=num_sequences,
        )
        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            fake_z = jnp.empty((1, z1_dim + z2_dim))
            self.fake_args_critic = (fake_z, fake_action(action_space))
        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            fake_feature_action = jnp.empty((1, num_sequences * feature_dim + (num_sequences - 1) * action_space.shape[0]))
            self.fake_args_actor = (fake_feature_action,)

        self.learning_step_model = 0
        self.learning_step_sac = 0
        self.num_sequences = num_sequences
        self.batch_size_sac = batch_size_sac
        self.batch_size_model = batch_size_model
        self.initial_learning_steps = initial_learning_steps

    def step(self, env, ob):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        state, reward, done, _ = env.step(action)
        ob.append(state, action)
        mask = self.get_mask(env, done)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            self.episode_step = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return None

    def select_action(self, ob):
        feature_action = self._preprocess(self.params_model, ob.state, ob.action)
        action = self._select_action(self.params_actor, feature_action)
        return np.array(action[0])

    def explore(self, ob):
        feature_action = self._preprocess(self.params_model, ob.state, ob.action)
        action = self._explore(self.params_actor, feature_action, next(self.rng))
        return np.array(action[0])

    def update(self, writer):
        NotImplementedError

    @abstractmethod
    def update_model(self, writer):
        pass

    @abstractmethod
    def update_sac(self, writer):
        pass

    def save_params(self, save_dir):
        save_params(self.params_model, os.path.join(save_dir, "params_model.npz"))

    def load_params(self, save_dir):
        self.params_model = load_params(os.path.join(save_dir, "params_model.npz"))
