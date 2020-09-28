import math

import haiku as hk
import jax.numpy as jnp
from jax import nn


class ContinuousVFunction(hk.Module):
    """
    Critic for PPO.
    """

    def __init__(
        self,
        num_critics=1,
        hidden_units=(64, 64),
        hidden_activation=jnp.tanh,
    ):
        super(ContinuousVFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        def v_func(x):
            for unit in self.hidden_units:
                x = hk.Linear(unit)(x)
                x = self.hidden_activation(x)
            return hk.Linear(1)(x)

        if self.num_critics == 1:
            return v_func(x)

        return [v_func(x) for _ in range(self.num_critics)]


class ContinuousQFunction(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units=(400, 300),
        hidden_activation=nn.relu,
    ):
        super(ContinuousQFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

    def __call__(self, x):
        def q_func(x):
            for unit in self.hidden_units:
                x = hk.Linear(unit)(x)
                x = self.hidden_activation(x)
            return hk.Linear(1)(x)

        if self.num_critics == 1:
            return q_func(x)

        return [q_func(x) for _ in range(self.num_critics)]


class DQNBody(hk.Module):
    """
    CNN for the atari environment.
    """

    def __init__(self):
        super(DQNBody, self).__init__()

    def __call__(self, x):
        # Floatify the image.
        x = x.astype(jnp.float32) / 255.0
        # Apply CNN.
        x = hk.Conv2D(32, kernel_shape=(8, 8), stride=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=(4, 4), stride=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=(3, 3), stride=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        # Flatten the feature map.
        return hk.Flatten()(x)


class DiscreteQFunction(hk.Module):
    """
    Critic for DQN and SAC-Discrete.
    """

    def __init__(
        self,
        action_dim,
        num_critics=1,
        hidden_units=(512,),
        hidden_activation=nn.relu,
        dueling_net=True,
    ):
        super(DiscreteQFunction, self).__init__()
        self.action_dim = action_dim
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.dueling_net = dueling_net

    def __call__(self, x):
        def q_func(x):
            a = x
            for unit in self.hidden_units:
                a = hk.Linear(unit)(a)
                a = self.hidden_activation(a)
            a = hk.Linear(self.action_dim)(a)
            if not self.dueling_net:
                return a

            v = x
            for unit in self.hidden_units:
                v = hk.Linear(unit)(v)
                v = self.hidden_activation(v)
            v = hk.Linear(1)(v)
            return a + v - a.mean(axis=1, keepdims=True)

        if self.num_critics == 1:
            return q_func(x)

        xs = [q_func(x) for _ in range(self.num_critics)]
        return xs


class DiscreteQuantileFunction(hk.Module):
    """
    Critic for QR-DQN and IQN.
    """

    def __init__(
        self,
        action_dim,
        num_critics=1,
        num_quantiles=200,
        hidden_units=(512,),
        hidden_activation=nn.relu,
        dueling_net=True,
    ):
        super(DiscreteQuantileFunction, self).__init__()
        self.action_dim = action_dim
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.dueling_net = dueling_net

    def __call__(self, x):
        def quantile_func(x):
            a = x
            for unit in self.hidden_units:
                a = hk.Linear(unit)(a)
                a = self.hidden_activation(a)
            a = hk.Linear(self.action_dim * self.num_quantiles)(a)
            a = a.reshape(-1, self.num_quantiles, self.action_dim)
            if not self.dueling_net:
                return a

            v = x
            for unit in self.hidden_units:
                v = hk.Linear(unit)(v)
                v = self.hidden_activation(v)
            v = hk.Linear(self.num_quantiles)(v)
            v = v.reshape(-1, self.num_quantiles, 1)
            return a + v - a.mean(axis=2, keepdims=True)

        if self.num_critics == 1:
            return quantile_func(x)

        xs = [quantile_func(x) for _ in range(self.num_critics)]
        return xs


class CosineEmbeddingNetwork(hk.Module):
    """
    Network for calculating cosine embeddings.
    """

    def __init__(
        self,
        state_dim,
        num_cosines=64,
    ):
        super(CosineEmbeddingNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_cosines = num_cosines

    def __call__(self, tau):
        pi = math.pi * jnp.arange(1, self.num_cosines + 1, dtype=jnp.float32).reshape(1, 1, self.num_cosines)
        cosine = jnp.cos(jnp.expand_dims(tau, 1) * pi).reshape(-1, self.num_cosines)
        return nn.relu(hk.Linear(self.state_dim)(cosine))
