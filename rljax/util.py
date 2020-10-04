import math
from functools import partial
from typing import Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def calculate_gaussian_log_prob(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of diagonal gaussian distributions.
    """
    return (-0.5 * jnp.square(noise) - log_std).sum(axis=1, keepdims=True) - 0.5 * jnp.log(2 * math.pi) * log_std.shape[1]


@jax.jit
def calculate_log_pi(
    log_std: jnp.ndarray,
    noise: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of the policies, which is diagonal gaussian distributions followed by tanh transformation.
    """
    return calculate_gaussian_log_prob(log_std, noise) - jnp.log(1 - jnp.square(action) + 1e-6).sum(axis=1, keepdims=True)


@jax.jit
def evaluate_lop_pi(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Calculate log probabilities of the policies given sampled actions.
    """
    noise = (jnp.arctanh(action) - mean) / (jnp.exp(log_std) + 1e-8)
    return calculate_log_pi(log_std, noise, action)


@jax.jit
def reparameterize_gaussian_with_tanh(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate stochastic actions and log probabilities.
    """
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, std.shape)
    action = jnp.tanh(mean + noise * std)
    return action, calculate_log_pi(log_std, noise, action)


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


def save_params(params, path):
    """
    Save parameters.
    """
    np.savez(path, **params)


def load_params(path):
    """
    Load parameters.
    """
    return hk.data_structures.to_immutable_dict(np.load(path))


@jax.jit
def add_noise(
    x: jnp.ndarray,
    key: jnp.ndarray,
    std: float,
    x_min: float,
    x_max: float,
) -> jnp.ndarray:
    """
    Add noise to actions.
    """
    return jnp.clip(x + jax.random.normal(key, x.shape) * std, x_min, x_max)


@jax.jit
def get_q_at_action(
    q_s: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get q values at (s, a).
    """

    def _get(q_s, action):
        return q_s[action]

    return jax.vmap(_get)(q_s, action)


@jax.jit
def get_quantile_at_action(
    quantile_s: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """
    Get quantile values at (s, a).
    """

    def _get(quantile_s, action):
        return quantile_s[:, action]

    return jax.vmap(_get)(quantile_s, action)


@jax.jit
def huber_fn(td: jnp.ndarray) -> jnp.ndarray:
    abs_td = jnp.abs(td)
    return jnp.where(abs_td <= 1.0, jnp.square(td), abs_td)


@partial(jax.jit, static_argnums=3)
def calculate_quantile_loss(
    td: jnp.ndarray,
    cum_p: jnp.ndarray,
    weight: jnp.ndarray,
    loss_type: float = "l2",
) -> jnp.ndarray:
    """
    Calculate quantile loss.
    """
    if loss_type == "l2":
        element_wise_loss = jnp.square(td)
    elif loss_type == "huber":
        element_wise_loss = huber_fn(td)
    else:
        NotImplementedError
    element_wise_loss *= jax.lax.stop_gradient(jnp.abs(cum_p[..., None] - (td < 0)))
    batch_loss = element_wise_loss.sum(axis=1).mean(axis=1, keepdims=True)
    return (batch_loss * weight).mean()


@partial(jax.jit, static_argnums=2)
def preprocess_state(
    state: np.ndarray,
    key: jnp.ndarray,
    bits: float = 5,
) -> jnp.ndarray:
    """
    Preprocess pixel states to fit into [-0.5, 0.5].
    """
    state = state.astype(jnp.float32)
    bins = 2 ** bits
    if bits < 8:
        state = jnp.floor(state / 2 ** (8 - bits))
    state = state / bins
    state = state + jax.random.uniform(key, state.shape) / bins
    state = state - 0.5
    return state
