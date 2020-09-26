from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from rljax.buffer.replay_buffer import ReplayBuffer
from rljax.buffer.segment_tree import MinTree, SumTree


def calculate_pa(
    error: jnp.ndarray,
    alpha: float,
    min_pa: float,
    max_pa: float,
    eps: float,
) -> jnp.ndarray:
    return jnp.clip((error + eps) ** alpha, min_pa, max_pa)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer.
    """

    def __init__(
        self,
        buffer_size,
        state_space,
        action_space,
        gamma,
        nstep,
        alpha=0.6,
        beta=0.4,
        beta_steps=10 ** 5,
        min_pa=0.0,
        max_pa=1.0,
        eps=0.01,
    ):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, state_space, action_space, gamma, nstep)

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1.0 - beta) / beta_steps
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        self._cached_idxes = None

        tree_size = 1
        while tree_size < buffer_size:
            tree_size *= 2
        self.tree_sum = SumTree(tree_size)
        self.tree_min = MinTree(tree_size)
        self._calculate_pa = jax.jit(partial(calculate_pa, alpha=alpha, min_pa=min_pa, max_pa=max_pa, eps=eps))

    def _append(self, state, action, reward, next_state, done):
        # Assign max priority when stored for the first time.
        self.tree_min[self._p] = self.max_pa
        self.tree_sum[self._p] = self.max_pa
        super()._append(state, action, reward, next_state, done)

    def _sample_idx(self, batch_size):
        total_pa = self.tree_sum.reduce(0, self._n)
        rand = np.random.rand(batch_size) * total_pa
        idxes = [self.tree_sum.find_prefixsum_idx(r) for r in rand]
        self.beta = min(1.0, self.beta + self.beta_diff)
        return idxes

    def sample(self, batch_size):
        assert self._cached_idxes is None, "Update priorities before sampling."

        self._cached_idxes = self._sample_idx(batch_size)
        weight = self._calculate_weight(self._cached_idxes)
        batch = self._sample(self._cached_idxes)
        return weight, batch

    def _calculate_weight(self, idxes):
        min_pa = self.tree_min.reduce()
        weight = [(self.tree_sum[i] / min_pa) ** -self.beta for i in idxes]
        weight = np.array(weight, dtype=np.float32)
        return np.expand_dims(weight, axis=1)

    def update_priority(self, error):
        assert self._cached_idxes is not None, "Sample batch before updating priorities."
        pa = np.array(self._calculate_pa(error), dtype=np.float32).flatten()
        for index, pa in zip(self._cached_idxes, pa):
            self.tree_sum[index] = pa
            self.tree_min[index] = pa
        self._cached_idxes = None
