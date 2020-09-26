import haiku as hk
import jax.numpy as jnp
from jax import nn


class DeterministicPolicy(hk.Module):
    """
    Policy for DDPG and TD3.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(400, 300),
        hidden_activation=nn.relu,
    ):
        super(DeterministicPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.action_dim)(x)
        return jnp.tanh(x)


class StateDependentGaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(256, 256),
        hidden_activation=nn.relu,
    ):
        super(StateDependentGaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(2 * self.action_dim)(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        return mean, log_std


class StateIndependentGaussianPolicy(hk.Module):
    """
    Policy for PPO.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(64, 64),
        hidden_activation=jnp.tanh,
    ):
        super(StateIndependentGaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit)(x)
            x = self.hidden_activation(x)
        mean = hk.Linear(self.action_dim)(x)
        log_std = hk.get_parameter("log_std", (1, self.action_dim), init=jnp.zeros)

        return mean, log_std


class CategoricalPolicy(hk.Module):
    """
    Policy for SAC-Discrete.
    """

    def __init__(
        self,
        action_dim,
        hidden_units=(256, 256),
        hidden_activation=nn.relu,
    ):
        super(CategoricalPolicy, self).__init__()
        self.action_dim = action_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.action_dim)(x)
        pi = nn.softmax(x, axis=1)
        return pi, jnp.log(pi + 1e-6)
