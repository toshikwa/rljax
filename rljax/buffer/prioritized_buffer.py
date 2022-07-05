from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from rljax.buffer.replay_buffer import ReplayBuffer
from rljax.buffer.segment_tree import SumTree


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
        self.eps = eps
        self._cached_idxes = None
        self.max_pa = 1.0

        tree_size = 1
        while tree_size < buffer_size:
            tree_size *= 2
        self.tree_sum = SumTree(tree_size)

    def _append(self, state, action, reward, next_state, done):
        # Assign max priority when stored for the first time.
        self.tree_sum[self._p] = self.max_pa
        super()._append(state, action, reward, next_state, done)

    def _sample_idx(self, batch_size):
        total_pa = self.tree_sum.get_total_sum()
        rand = np.random.rand(batch_size) * total_pa
        idxes = [self.tree_sum.find_prefixsum_idx(r) for r in rand]
        self.beta = min(1.0, self.beta + self.beta_diff)
        return idxes

    def sample(self, batch_size):
        assert self._cached_idxes is None, "Update priorities before sampling."

        self._cached_idxes = self._sample_idx(batch_size)
        total_pa = self.tree_sum.get_total_sum()
        weight = self._calculate_weight(self._cached_idxes, total_pa)
        batch = self._sample(self._cached_idxes)
        return weight, batch

    def _calculate_weight(self, idxes, total_pa):
        weight = [((self.tree_sum[i]/total_pa) * self._n) ** -self.beta for i in idxes]
        weight = np.array(weight, dtype=np.float32)
        weight = weight/np.max(weight)
        return np.expand_dims(weight, axis=1)

    def update_priority(self, abs_td):
        assert self._cached_idxes is not None, "Sample batch before updating priorities."
        assert abs_td.shape[1:] == (1,)
        pa = np.array(self._calculate_pa(abs_td), dtype=np.float32).flatten()
        for i, idx in enumerate(self._cached_idxes):
            self.tree_sum[idx] = pa[i]
            if pa[i] > self.max_pa:
                self.max_pa = pa[i]
        self._cached_idxes = None

    @partial(jax.jit, static_argnums=0)
    def _calculate_pa(self, abs_td: jnp.ndarray) -> jnp.ndarray:
        return (abs_td + self.eps) ** self.alpha
