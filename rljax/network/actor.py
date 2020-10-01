import haiku as hk
import jax.numpy as jnp
from jax import nn

from rljax.network.base import MLP, DQNBody


class DeterministicPolicy(hk.Module):
    """
    Policy for DDPG and TD3.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        hidden_activation=nn.relu,
    ):
        super(DeterministicPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        return MLP(
            self.action_space.shape[0],
            self.hidden_units,
            self.hidden_activation,
            output_activation=jnp.tanh,
        )(x)


class StateDependentGaussianPolicy(hk.Module):
    """
    Policy for SAC.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(256, 256),
        hidden_activation=nn.relu,
    ):
        super(StateDependentGaussianPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        x = MLP(2 * self.action_space.shape[0], self.hidden_units, self.hidden_activation)(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        return mean, jnp.clip(log_std, -20, 2)


class StateIndependentGaussianPolicy(hk.Module):
    """
    Policy for PPO.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(64, 64),
        hidden_activation=jnp.tanh,
    ):
        super(StateIndependentGaussianPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        mean = MLP(
            self.action_space.shape[0],
            self.hidden_units,
            self.hidden_activation,
            output_scale=0.01,
        )(x)
        log_std = hk.get_parameter("log_std", (1, self.action_space.shape[0]), init=jnp.zeros)
        return mean, jnp.clip(log_std, -20, 2)


class CategoricalPolicy(hk.Module):
    """
    Policy for SAC-Discrete.
    """

    def __init__(
        self,
        action_space,
        hidden_units=(512,),
        hidden_activation=nn.relu,
    ):
        super(CategoricalPolicy, self).__init__()
        self.action_space = action_space
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        if len(x.shape) == 4:
            x = DQNBody()(x)
        x = MLP(self.action_space.n, self.hidden_units, self.hidden_activation)(x)
        pi = nn.softmax(x, axis=1)
        log_pi = jnp.log(pi + (pi == 0.0) * 1e-8)
        return pi, log_pi
