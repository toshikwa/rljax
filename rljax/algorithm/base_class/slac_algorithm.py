import os
from abc import abstractmethod

import numpy as np

from rljax.algorithm.base_class.base_algoirithm import OffPolicyAlgorithm
from rljax.buffer import SLACReplayBuffer
from rljax.util import load_params, save_params


class SlacAlgorithm(OffPolicyAlgorithm):
    """
    Base class for SLAC-based algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        num_sequences,
        num_critics,
        buffer_size,
        batch_size_sac,
        batch_size_model,
        start_steps,
        initial_learning_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        self.buffer = SLACReplayBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            num_sequences=num_sequences,
        )
        super(SlacAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=1,
            buffer_size=buffer_size,
            use_per=False,
            batch_size=None,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
            tau=tau,
        )

        self.learning_step_model = 0
        self.learning_step_sac = 0
        self.num_sequences = num_sequences
        self.num_critics = num_critics
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
        action = self._explore(self.params_actor, next(self.rng), feature_action)[0]
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
