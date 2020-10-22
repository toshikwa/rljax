from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten


@jax.jit
def clip_gradient(
    grad: Any,
    max_grad_norm: float,
) -> Any:
    """
    Clip gradients.
    """
    return jax.tree_map(lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm), grad)


@jax.jit
def soft_update(
    target_params: hk.Params,
    online_params: hk.Params,
    tau: float,
) -> hk.Params:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    return jax.tree_multimap(lambda t, s: (1 - tau) * t + tau * s, target_params, online_params)


@jax.jit
def weight_decay(params: hk.Params) -> jnp.ndarray:
    """
    Calculate the sum of L2 norms of parameters.
    """
    leaves, _ = tree_flatten(params)
    return 0.5 * sum(jnp.vdot(x, x) for x in leaves)
