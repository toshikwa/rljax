from abc import ABC, abstractmethod

from haiku import PRNGSequence
from rljax.common.buffer import ReplayBuffer


class Algorithm(ABC):
    """
    Base class for algorithms.
    """

    def __init__(self, state_shape, action_shape, seed, gamma):
        self.rng = PRNGSequence(seed)
        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def explore(self, state):
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

    def __init__(self, state_shape, action_shape, seed, gamma, buffer_size, batch_size, start_steps, tau):
        super(ContinuousOffPolicyAlgorithm, self).__init__(state_shape, action_shape, seed, gamma)

        self.buffer = ReplayBuffer(buffer_size=buffer_size, state_shape=state_shape, action_shape=action_shape)
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.tau = tau

    def is_update(self, step):
        return step >= max(self.start_steps, self.batch_size)

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
