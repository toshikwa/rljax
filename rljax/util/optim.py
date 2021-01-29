from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_flatten


@partial(jax.jit, static_argnums=(0, 1, 4))
def optimize(
    fn_loss: Any,
    opt: Any,
    opt_state: Any,
    params_to_update: hk.Params,
    max_grad_norm: float or None,
    *args,
    **kwargs,
) -> Tuple[Any, hk.Params, jnp.ndarray, Any]:
    (loss, aux), grad = jax.value_and_grad(fn_loss, has_aux=True)(
        params_to_update,
        *args,
        **kwargs,
    )
    if max_grad_norm is not None:
        grad = clip_gradient_norm(grad, max_grad_norm)
    update, opt_state = opt(grad, opt_state)
    params_to_update = optax.apply_updates(params_to_update, update)
    return opt_state, params_to_update, loss, aux


@jax.jit
def clip_gradient(
    grad: Any,
    max_value: float,
) -> Any:
    """
    Clip gradients.
    """
    return jax.tree_map(lambda g: jnp.clip(g, -max_value, max_value), grad)


@jax.jit
def clip_gradient_norm(
    grad: Any,
    max_grad_norm: float,
) -> Any:
    """
    Clip norms of gradients.
    """

    def _clip_gradient_norm(g):
        clip_coef = max_grad_norm / (jax.lax.stop_gradient(jnp.linalg.norm(g)) + 1e-6)
        clip_coef = jnp.clip(clip_coef, a_max=1.0)
        return g * clip_coef

    return jax.tree_map(lambda g: _clip_gradient_norm(g), grad)


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
