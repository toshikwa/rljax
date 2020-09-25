from abc import ABC, abstractmethod

from haiku import PRNGSequence


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
