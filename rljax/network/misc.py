from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from rljax.network.base import MLP


class CumProbNetwork(hk.Module):
    """
    Fraction Proposal Network for FQF.
    """

    def __init__(self, num_quantiles=64):
        super(CumProbNetwork, self).__init__()
        self.num_quantiles = num_quantiles

    def __call__(self, x):
        w_init = hk.initializers.Orthogonal(scale=1.0 / np.sqrt(3.0))
        p = nn.softmax(hk.Linear(self.num_quantiles, w_init=w_init)(x))
        cum_p = jnp.concatenate([jnp.zeros((p.shape[0], 1)), jnp.cumsum(p, axis=1)], axis=1)
        cum_p_prime = (cum_p[:, 1:] + cum_p[:, :-1]) / 2.0
        return cum_p, cum_p_prime


class SACLinear(hk.Module):
    """
    Linear layer for SAC+AE.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def __call__(self, x):
        w_init = hk.initializers.Orthogonal(scale=1.0)
        x = hk.Linear(self.feature_dim, w_init=w_init)(x)
        x = hk.LayerNorm(axis=1, create_scale=True, create_offset=True)(x)
        x = jnp.tanh(x)
        return x


class ConstantGaussian(hk.Module):
    """
    Constant diagonal gaussian distribution for SLAC.
    """

    def __init__(self, output_dim, std):
        super().__init__()
        self.output_dim = output_dim
        self.std = std

    def __call__(self, x):
        mean = jnp.zeros((x.shape[0], self.output_dim))
        std = jnp.ones((x.shape[0], self.output_dim)) * self.std
        return jax.lax.stop_gradient(mean), jax.lax.stop_gradient(std)


class Gaussian(hk.Module):
    """
    Diagonal gaussian distribution with state dependent variances for SLAC.
    """

    def __init__(self, output_dim, hidden_units=(256, 256), negative_slope=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.negative_slope = negative_slope

    def __call__(self, x):
        x = MLP(
            output_dim=2 * self.output_dim,
            hidden_units=self.hidden_units,
            hidden_activation=partial(nn.leaky_relu, negative_slope=self.negative_slope),
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        std = nn.softplus(log_std) + 1e-5
        return mean, std
