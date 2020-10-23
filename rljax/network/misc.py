from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from rljax.network.base import MLP
from rljax.network.initializer import DeltaOrthogonal


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


class SACEncoder(hk.Module):
    """
    Encoder for SAC+AE.
    """

    def __init__(self, num_layers=4, num_filters=32, negative_slope=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.negative_slope = negative_slope

    def __call__(self, x):
        # Floatify the image.
        x = x.astype(jnp.float32) / 255.0

        # Apply CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        x = hk.Conv2D(self.num_filters, kernel_shape=4, stride=2, padding="VALID", w_init=w_init)(x)
        x = nn.leaky_relu(x, self.negative_slope)
        for _ in range(self.num_layers - 1):
            x = hk.Conv2D(self.num_filters, kernel_shape=3, stride=1, padding="VALID", w_init=w_init)(x)
            x = nn.leaky_relu(x, self.negative_slope)
        # Flatten the feature map.
        return hk.Flatten()(x)


class SACDecoder(hk.Module):
    """
    Decoder for SAC+AE.
    """

    def __init__(self, state_space, num_layers=4, num_filters=32, negative_slope=0.1):
        super().__init__()
        self.state_space = state_space
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.negative_slope = negative_slope
        self.map_size = 43 - 2 * num_layers
        self.last_conv_dim = num_filters * self.map_size * self.map_size

    def __call__(self, x):
        # Apply linear layer.
        w_init = hk.initializers.Orthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        x = hk.Linear(self.last_conv_dim, w_init=w_init)(x)
        x = nn.leaky_relu(x, self.negative_slope).reshape(-1, self.map_size, self.map_size, self.num_filters)

        # Apply Transposed CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        for _ in range(self.num_layers - 1):
            x = hk.Conv2DTranspose(self.num_filters, kernel_shape=3, stride=1, padding="VALID", w_init=w_init)(x)
            x = nn.leaky_relu(x, self.negative_slope)

        # Apply output layer.
        w_init = DeltaOrthogonal(scale=1.0)
        x = hk.Conv2DTranspose(self.state_space.shape[2], kernel_shape=4, stride=2, padding="VALID", w_init=w_init)(x)
        return x


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
            hidden_scale=np.sqrt(2 / (1 + self.negative_slope ** 2)),
        )(x)
        mean, log_std = jnp.split(x, 2, axis=1)
        std = nn.softplus(log_std) + 1e-5
        return mean, std


class SLACEncoder(hk.Module):
    """
    Encoder for SLAC.
    """

    def __init__(self, output_dim=256, negative_slope=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.negative_slope = negative_slope

    def __call__(self, x):
        B, S, H, W, C = x.shape

        # Floatify the image.
        x = x.astype(jnp.float32) / 255.0
        # Reshape.
        x = x.reshape([B * S, H, W, C])
        # Apply CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        depth = [32, 64, 128, 256, self.output_dim]
        kernel = [5, 3, 3, 3, 4]
        stride = [2, 2, 2, 2, 1]
        padding = ["SAME", "SAME", "SAME", "SAME", "VALID"]

        for i in range(5):
            x = hk.Conv2D(
                depth[i],
                kernel_shape=kernel[i],
                stride=stride[i],
                padding=padding[i],
                w_init=w_init,
            )(x)
            x = nn.leaky_relu(x, self.negative_slope)

        return x.reshape([B, S, -1])


class SLACDecoder(hk.Module):
    """
    Decoder for SLAC.
    """

    def __init__(self, state_space, std=1.0, negative_slope=0.2):
        super().__init__()
        self.state_space = state_space
        self.std = std
        self.negative_slope = negative_slope

    def __call__(self, x):
        B, S, latent_dim = x.shape

        # Reshape.
        x = x.reshape([B * S, 1, 1, latent_dim])

        # Apply CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        depth = [256, 128, 64, 32, self.state_space.shape[2]]
        kernel = [4, 3, 3, 3, 5]
        stride = [1, 2, 2, 2, 2]
        padding = ["VALID", "SAME", "SAME", "SAME", "SAME"]

        for i in range(4):
            x = hk.Conv2DTranspose(
                depth[i],
                kernel_shape=kernel[i],
                stride=stride[i],
                padding=padding[i],
                w_init=w_init,
            )(x)
            x = nn.leaky_relu(x, self.negative_slope)

        w_init = DeltaOrthogonal(scale=1.0)
        x = hk.Conv2DTranspose(
            depth[-1],
            kernel_shape=kernel[-1],
            stride=stride[-1],
            padding=padding[-1],
            w_init=w_init,
        )(x)

        _, W, H, C = x.shape
        x = x.reshape([B, S, W, H, C])
        return x, jax.lax.stop_gradient(jnp.ones_like(x) * self.std)
