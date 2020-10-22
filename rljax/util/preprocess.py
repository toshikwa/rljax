import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def add_noise(
    x: jnp.ndarray,
    key: jnp.ndarray,
    std: float,
    out_min: float = -np.inf,
    out_max: float = np.inf,
    noise_min: float = -np.inf,
    noise_max: float = np.inf,
) -> jnp.ndarray:
    """
    Add noise to actions.
    """
    noise = jnp.clip(jax.random.normal(key, x.shape), noise_min, noise_max)
    return jnp.clip(x + noise * std, out_min, out_max)


@jax.jit
def preprocess_state(
    state: np.ndarray,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """
    Preprocess pixel states to fit into [-0.5, 0.5].
    """
    state = state.astype(jnp.float32)
    state = jnp.floor(state / 8)
    state = state / 32
    state = state + jax.random.uniform(key, state.shape) / 32
    state = state - 0.5
    return state


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
