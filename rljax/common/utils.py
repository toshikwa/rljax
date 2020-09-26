import math

import flax
import jax
import jax.numpy as jnp


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
):
    """
    Calculate log probabilities of the policies given sampled actions.
    """
    noise = (jnp.arctanh(action) - mean) / (jnp.exp(log_std) + 1e-8)
    return calculate_log_pi(log_std, noise, action)


@jax.jit
def clip_gradient(
    grad: flax.nn.Model,
    max_grad_norm: float,
) -> flax.nn.Model:
    """
    Clip gradients.
    """
    params = jax.tree_multimap(lambda g: jnp.clip(g, -max_grad_norm, max_grad_norm), grad.params)
    return grad.replace(params=params)


@jax.jit
def soft_update(
    target: flax.nn.Model,
    source: flax.nn.Model,
    tau: float,
) -> flax.nn.Model:
    """
    Update target network using Polyak-Ruppert Averaging.
    """
    params = jax.tree_multimap(lambda t, s: tau * s + (1 - tau) * t, target.params, source.params)
    return target.replace(params=params)


@jax.jit
def update_network(
    optim: flax.optim.Optimizer,
    grad: flax.nn.Model,
) -> flax.optim.Optimizer:
    """
    Update the network with gradient information.
    """
    return optim.apply_gradient(grad)


@jax.jit
def add_noise(
    x: jnp.ndarray,
    key: jnp.ndarray,
    std: float,
    x_min: float,
    x_max: float,
):
    """
    Add noise to actions.
    """
    return jnp.clip(x + jax.random.normal(key, x.shape), x_min, x_max)
