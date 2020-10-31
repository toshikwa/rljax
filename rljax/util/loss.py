from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def huber(td: jnp.ndarray) -> jnp.ndarray:
    """ Huber function. """
    abs_td = jnp.abs(td)
    return jnp.where(abs_td <= 1.0, jnp.square(td), abs_td)


@partial(jax.jit, static_argnums=3)
def quantile_loss(
    td: jnp.ndarray,
    cum_p: jnp.ndarray,
    weight: jnp.ndarray,
    loss_type: str,
) -> jnp.ndarray:
    """
    Calculate quantile loss.
    """
    if loss_type == "l2":
        element_wise_loss = jnp.square(td)
    elif loss_type == "huber":
        element_wise_loss = huber(td)
    else:
        NotImplementedError
    element_wise_loss *= jax.lax.stop_gradient(jnp.abs(cum_p[..., None] - (td < 0)))
    batch_loss = element_wise_loss.sum(axis=1).mean(axis=1, keepdims=True)
    return (batch_loss * weight).mean()
