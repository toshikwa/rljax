import math

import jax
import jax.numpy as jnp


@jax.jit
def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * jnp.square(noise) - log_std).sum(axis=1, keepdims=True) - 0.5 * jnp.log(2 * math.pi) * log_std.shape[1]


@jax.jit
def calculate_log_pi(log_std, noise, action):
    return calculate_gaussian_log_prob(log_std, noise) - jnp.log(1 - jnp.square(action) + 1e-6).sum(axis=1, keepdims=True)


@jax.jit
def soft_update(target, source, tau):
    params = jax.tree_multimap(lambda m1, mt: tau * m1 + (1 - tau) * mt, source.params, target.params)
    return target.replace(params=params)
