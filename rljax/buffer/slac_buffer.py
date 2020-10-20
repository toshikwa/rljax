from collections import deque

import numpy as np
from gym.spaces import Box, Discrete


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self.reset()

    def reset(self):
        self._reset_episode = False
        self.state = deque(maxlen=self.num_sequences + 1)
        self.action = deque(maxlen=self.num_sequences)
        self.reward = deque(maxlen=self.num_sequences)
        self.done = deque(maxlen=self.num_sequences)

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state.append(state)

    def append(self, action, reward, done, next_state):
        assert self._reset_episode
        self.action.append(action)
        self.reward.append([reward])
        self.done.append([done])
        self.state.append(next_state)

    def get(self):
        state = LazyFrames(self.state)
        action = np.array(self.action, dtype=np.float32)
        reward = np.array(self.reward, dtype=np.float32)
        done = np.array(self.done, dtype=np.float32)
        return state, action, reward, done

    def is_empty(self):
        return len(self.reward) == 0

    def is_full(self):
        return len(self.reward) == self.num_sequences

    def __len__(self):
        return len(self.reward)


class SLACReplayBuffer:
    """
    Replay Buffer for SLAC.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        num_sequences,
    ):
        assert len(state_space.shape) in (1, 3)

        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_space.shape
        self.use_image = len(self.state_shape) == 3

        if self.use_image:
            # Store images as a list of LazyFrames, which uses 4 times less memory.
            self.state = [None] * buffer_size
        else:
            self.state = np.empty((buffer_size, num_sequences + 1, *state_space.shape), dtype=np.float32)

        if type(action_space) == Box:
            self.action = np.empty((buffer_size, num_sequences, *action_space.shape), dtype=np.float32)
        elif type(action_space) == Discrete:
            self.action = np.empty((buffer_size, num_sequences, 1), dtype=np.int32)
        else:
            NotImplementedError

        self.reward = np.empty((buffer_size, num_sequences, 1), dtype=np.float32)
        self.done = np.empty((buffer_size, num_sequences, 1), dtype=np.float32)

        # Buffer to store a sequence of trajectories.
        self.seq_buffer = SequenceBuffer(num_sequences)

    def reset_episode(self, state):
        """
        Reset the sequence buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.seq_buffer.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done=None):
        self.seq_buffer.append(action, reward, done, next_state)

        if self.seq_buffer.is_full():
            state, action, reward, done = self.seq_buffer.get()
            self._append(state, action, reward, done)

        if episode_done:
            self.seq_buffer.reset()

    def _append(self, state, action, reward, done):
        self.state[self._p] = state
        self.action[self._p] = action
        self.reward[self._p] = reward
        self.done[self._p] = done

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def _sample_idx(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample_state(self, idxes):
        if self.use_image:
            state = np.empty((len(idxes), self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
            for i, idx in enumerate(idxes):
                state[i, ...] = self.state[idx]
        else:
            state = self.state[idxes]
        return state

    def sample_latent(self, batch_size):
        idxes = self._sample_idx(batch_size)
        return (self._sample_state(idxes), self.action[idxes], self.reward[idxes], self.done[idxes])

    def sample_sac(self, batch_size):
        idxes = self._sample_idx(batch_size)
        return (self._sample_state(idxes), self.action[idxes], self.reward[idxes, -1], self.done[idxes, -1])
