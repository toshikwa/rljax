from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rljax.buffer.replay_buffer import ReplayBuffer
from rljax.buffer.segment_tree import MinTree, SumTree


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
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            gamma=gamma,
            nstep=nstep,
        )

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
        min_pa = self.tree_min.reduce(0, self._n)
        weight = [(self.tree_sum[i] / min_pa) ** -self.beta for i in idxes]
        weight = np.array(weight, dtype=np.float32)
        return np.expand_dims(weight, axis=1)

    def update_priority(self, abs_td):
        assert self._cached_idxes is not None, "Sample batch before updating priorities."
        assert abs_td.shape[1:] == (1,)
        pa = np.array(self._calculate_pa(abs_td), dtype=np.float32).flatten()
        for i, idx in enumerate(self._cached_idxes):
            self.tree_sum[idx] = pa[i]
            self.tree_min[idx] = pa[i]
        self._cached_idxes = None

    @partial(jax.jit, static_argnums=0)
    def _calculate_pa(self, abs_td: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip((abs_td + self.eps) ** self.alpha, self.min_pa, self.max_pa)
