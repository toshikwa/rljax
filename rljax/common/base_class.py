from abc import ABC, abstractmethod

import numpy as np

from haiku import PRNGSequence
from rljax.common.buffer import ReplayBuffer


class Algorithm(ABC):
    """
    Base class for algorithms.
    """

    def __init__(self, state_space, action_space, seed, gamma, buffer_size):
        self.rng = PRNGSequence(seed)
        self.buffer = ReplayBuffer(buffer_size=buffer_size, state_space=state_space, action_space=action_space)
        self.learning_steps = 0
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def step(self, env, state, t, step):
        pass

    @abstractmethod
    def update(self):
        pass


class ContinuousOffPolicyAlgorithm(Algorithm):
    """
    Base class for continuous off-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        buffer_size,
        batch_size,
        start_steps,
        tau,
    ):
        super(ContinuousOffPolicyAlgorithm, self).__init__(state_space, action_space, seed, gamma, buffer_size)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau

    def is_update(self, step):
        return step >= self.start_steps

    def step(self, env, state, t, step):
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t


class DiscreteOffPolicyAlgorithm(Algorithm):
    """
    Base class for discrete off-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        buffer_size,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target,
    ):
        super(DiscreteOffPolicyAlgorithm, self).__init__(state_space, action_space, seed, gamma, buffer_size)

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target

    def is_update(self, step):
        return step % self.update_interval == 0 and step >= self.start_steps
