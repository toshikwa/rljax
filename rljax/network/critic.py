import math

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn

from rljax.network.base import MLP
from rljax.network.conv import DQNBody


class ContinuousVFunction(hk.Module):
    """
    Critic for PPO.
    """

    def __init__(
        self,
        num_critics=1,
        hidden_units=(64, 64),
    ):
        super(ContinuousVFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units

    def __call__(self, x):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=jnp.tanh,
            )(x)

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQFunction(hk.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def __init__(
        self,
        num_critics=2,
        hidden_units=(256, 256),
        d2rl=False,
    ):
        super(ContinuousQFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                1,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        # Return list even if num_critics == 1 for simple implementation.
        return [_fn(x) for _ in range(self.num_critics)]


class ContinuousQuantileFunction(hk.Module):
    """
    Critic for TQC.
    """

    def __init__(
        self,
        num_critics=5,
        hidden_units=(512, 512, 512),
        num_quantiles=25,
        d2rl=False,
    ):
        super(ContinuousQuantileFunction, self).__init__()
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.num_quantiles = num_quantiles
        self.d2rl = d2rl

    def __call__(self, s, a):
        def _fn(x):
            return MLP(
                self.num_quantiles,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
                d2rl=self.d2rl,
            )(x)

        x = jnp.concatenate([s, a], axis=1)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteQFunction(hk.Module):
    """
    Critic for DQN and SAC-Discrete.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        hidden_units=(512,),
        dueling_net=False,
    ):
        super(DiscreteQFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net

    def __call__(self, x):
        def _fn(x):
            if len(x.shape) == 4:
                x = DQNBody()(x)
            output = MLP(
                self.action_space.n,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            if self.dueling_net:
                baseline = MLP(
                    1,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                return output + baseline - output.mean(axis=1, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteQuantileFunction(hk.Module):
    """
    Critic for QR-DQN.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        num_quantiles=200,
        hidden_units=(512,),
        dueling_net=True,
    ):
        super(DiscreteQuantileFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.num_quantiles = num_quantiles
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net

    def __call__(self, x):
        def _fn(x):
            if len(x.shape) == 4:
                x = DQNBody()(x)
            output = MLP(
                self.action_space.n * self.num_quantiles,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            output = output.reshape(-1, self.num_quantiles, self.action_space.n)
            if self.dueling_net:
                baseline = MLP(
                    self.num_quantiles,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                baseline = baseline.reshape(-1, self.num_quantiles, 1)
                return output + baseline - output.mean(axis=2, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x)
        return [_fn(x) for _ in range(self.num_critics)]


class DiscreteImplicitQuantileFunction(hk.Module):
    """
    Critic for IQN and FQF.
    """

    def __init__(
        self,
        action_space,
        num_critics=1,
        num_cosines=64,
        hidden_units=(512,),
        dueling_net=True,
    ):
        super(DiscreteImplicitQuantileFunction, self).__init__()
        self.action_space = action_space
        self.num_critics = num_critics
        self.num_cosines = num_cosines
        self.hidden_units = hidden_units
        self.dueling_net = dueling_net
        self.pi = math.pi * jnp.arange(1, num_cosines + 1, dtype=jnp.float32).reshape(1, 1, num_cosines)

    def __call__(self, x, cum_p):
        def _fn(x, cum_p):
            if len(x.shape) == 4:
                x = DQNBody()(x)

            # NOTE: For IQN and FQF, number of quantiles are variable.
            feature_dim = x.shape[1]
            num_quantiles = cum_p.shape[1]
            # Calculate features.
            cosine = jnp.cos(jnp.expand_dims(cum_p, 2) * self.pi).reshape(-1, self.num_cosines)
            cosine_feature = nn.relu(hk.Linear(feature_dim)(cosine)).reshape(-1, num_quantiles, feature_dim)
            x = (x.reshape(-1, 1, feature_dim) * cosine_feature).reshape(-1, feature_dim)
            # Apply quantile network.
            output = MLP(
                self.action_space.n,
                self.hidden_units,
                hidden_activation=nn.relu,
                hidden_scale=np.sqrt(2),
            )(x)
            output = output.reshape(-1, num_quantiles, self.action_space.n)
            if self.dueling_net:
                baseline = MLP(
                    1,
                    self.hidden_units,
                    hidden_activation=nn.relu,
                    hidden_scale=np.sqrt(2),
                )(x)
                baseline = baseline.reshape(-1, num_quantiles, 1)
                return output + baseline - output.mean(axis=2, keepdims=True)
            else:
                return output

        if self.num_critics == 1:
            return _fn(x, cum_p)
        return [_fn(x, cum_p) for _ in range(self.num_critics)]
