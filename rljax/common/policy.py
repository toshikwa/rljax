import jax
import jax.numpy as jnp
from flax import nn
from rljax.common.utils import calculate_log_pi


class DeterministicPolicy(nn.Module):
    """
    Policy for DDPG and TD3.
    """

    def apply(self, x, action_dim, hidden_units=(400, 300), hidden_activation=nn.relu):
        for unit in hidden_units:
            x = nn.Dense(x, features=unit)
            x = hidden_activation(x)
        x = nn.Dense(x, features=action_dim)
        return jnp.tanh(x)


class StateDependentGaussianPolicy(nn.Module):
    """
    Policy for SAC.
    """

    def apply(self, x, action_dim, key=None, deterministic=True, hidden_units=(256, 256), hidden_activation=nn.relu):
        for unit in hidden_units:
            x = nn.Dense(x, features=unit)
            x = hidden_activation(x)
        x = nn.Dense(x, features=2 * action_dim)
        mean, log_std = jnp.split(x, 2, axis=1)

        if deterministic:
            return jnp.tanh(mean)

        else:
            log_std = jnp.clip(log_std, -20, 2)
            std = jnp.exp(log_std)
            noise = jax.random.normal(key, std.shape)
            action = jnp.tanh(mean + noise * std)
            log_pi = calculate_log_pi(log_std, noise, action)
            return action, log_pi
