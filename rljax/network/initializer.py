from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp


class DeltaOrthogonal(hk.initializers.Initializer):
    """
    Delta-orthogonal initializer.
    """

    def __init__(self, scale=1.0, axis=-1):
        self.scale = scale
        self.axis = axis

    def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
        if len(shape) not in [3, 4, 5]:
            raise ValueError("Delta orthogonal initializer requires 3D, 4D or 5D shape.")
        w_mat = jnp.zeros(shape, dtype=dtype)
        w_orthogonal = hk.initializers.Orthogonal(self.scale, self.axis)(shape[-2:], dtype)
        if len(shape) == 3:
            k = shape[0]
            return jax.ops.index_update(
                w_mat,
                jax.ops.index[(k - 1) // 2, ...],
                w_orthogonal,
            )
        elif len(shape) == 4:
            k1, k2 = shape[:2]
            return jax.ops.index_update(
                w_mat,
                jax.ops.index[(k1 - 1) // 2, (k2 - 1) // 2, ...],
                w_orthogonal,
            )
        else:
            k1, k2, k3 = shape[:3]
            return jax.ops.index_update(
                w_mat,
                jax.ops.index[(k1 - 1) // 2, (k2 - 1) // 2, (k3 - 1) // 2, ...],
                w_orthogonal,
            )
