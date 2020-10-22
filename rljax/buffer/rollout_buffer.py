import numpy as np
from gym.spaces import Box, Discrete


class RolloutBuffer:
    """
    Rollout Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size

        self.state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)
        self.reward = np.empty((buffer_size, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, 1), dtype=np.float32)
        self.log_pi = np.empty((buffer_size, 1), dtype=np.float32)
        self.next_state = np.empty((buffer_size, *state_space.shape), dtype=np.float32)

        if type(action_space) == Box:
            self.action = np.empty((buffer_size, *action_space.shape), dtype=np.float32)
        elif type(action_space) == Discrete:
            self.action = np.empty((buffer_size, 1), dtype=np.int32)
        else:
            NotImplementedError

    def append(self, state, action, reward, done, log_pi, next_state):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = float(reward)
        self.done[self._p] = float(done)
        self.log_pi[self._p] = float(log_pi)
        self.next_state[self._p] = next_state

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def get(self):
        return (
            self.state,
            self.action,
            self.reward,
            self.done,
            self.log_pi,
            self.next_state,
        )
