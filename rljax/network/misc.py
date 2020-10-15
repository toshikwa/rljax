import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn

from .initializer import DeltaOrthogonal


class CumProbNetwork(hk.Module):
    """
    Fraction Proposal Network for FQF.
    """

    def __init__(self, num_quantiles=64):
        super(CumProbNetwork, self).__init__()
        self.num_quantiles = num_quantiles

    def __call__(self, x):
        w_init = hk.initializers.VarianceScaling(scale=1.0 / np.sqrt(3.0), distribution="uniform")
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
