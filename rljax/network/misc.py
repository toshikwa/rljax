from functools import partial

import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn

from rljax.network.base import MLP
from rljax.network.initializer import DeltaOrthogonal
from rljax.util import reparameterize_gaussian


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
        return mean, std


class Gaussian(hk.Module):
    """
    Diagonal gaussian distribution with state dependent variances for SLAC.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256), negative_slope=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.negative_slope = negative_slope

    def __call__(self, x):
        x = MLP(
            output_dim=2 * self.output_dim,
            hidden_units=self.hidden_units,
            hidden_activation=partial(nn.leaky_relu, negative_slope=self.negative_slope),
        )(x)
        mean, std = jnp.split(x, 2, axis=1)
        std = nn.softplus(std) + 1e-5
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

        # Reshape.
        x = x.reshape([B * S, H, W, C])

        # Apply CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        depth = [32, 64, 128, 256, self.output_dim]
        kernel = [5, 3, 3, 3, 4]
        stride = [2, 2, 2, 2, 1]
        padding = ["SAME", "SAME", "SAME", "SAME", "VALID"]

        for i in range(5):
            x = hk.Conv2DTranspose(
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

    def __init__(self, state_space, input_dim=288, std=1.0, negative_slope=0.2):
        super().__init__()
        self.state_space = state_space
        self.input_dim = input_dim
        self.std = std
        self.negative_slope = negative_slope

    def __call__(self, x):
        B, S, latent_dim = x.shape

        # Reshape.
        x = x.reshape([B * S, latent_dim, 1, 1])

        # Apply CNN.
        w_init = DeltaOrthogonal(scale=np.sqrt(2 / (1 + self.negative_slope ** 2)))
        depth = [256, 128, 64, 32, self.state_space.shape[2]]
        kernel = [4, 3, 3, 3, 5]
        stride = [1, 2, 2, 2, 2]
        padding = ["VALID", "SAME", "SAME", "SAME", "SAME"]

        for i in range(4):
            x = hk.Conv2D(
                depth[i],
                kernel_shape=kernel[i],
                stride=stride[i],
                padding=padding[i],
                w_init=w_init,
            )(x)
            x = nn.leaky_relu(x, self.negative_slope)

        w_init = DeltaOrthogonal(scale=1.0)
        x = hk.Conv2D(
            depth[-1],
            kernel_shape=kernel[-1],
            stride=stride[-1],
            padding=padding[-1],
            w_init=w_init,
        )(x)

        _, W, H, C = x.shape
        x = x.reshape([B, S, W, H, C])
        return x, jnp.ones_like(x) * self.std


class StochasticLatentVariablePrior(hk.Module):
    """
    Prior distribution of stochastic latent variable model for SLAC.
    """

    def __init__(
        self,
        action_space,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super().__init__()
        self.action_space = action_space
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.hidden_units = hidden_units

    def __call__(self, action_):
        # p(z1(0)) = N(0, I)
        z1_prior_init = ConstantGaussian(self.z1_dim, 1.0)
        # p(z2(0) | z1(0))
        z2_prior_init = Gaussian(self.z1_dim, self.z2_dim, self.hidden_units)
        # p(z1(t+1) | z2(t), a(t))
        z1_prior = Gaussian(self.z2_dim + self.action_space.shape[0], self.z1_dim, self.hidden_units)
        # p(z2(t+1) | z1(t+1), z2(t), a(t))
        z2_prior = Gaussian(self.z1_dim + self.z2_dim + self.action_space.shape[0], self.z2_dim, self.hidden_units)

        z1_mean_ = []
        z1_std_ = []
        z1_ = []
        z2_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = z1_prior_init(action_[:, 0])
        z1 = reparameterize_gaussian(z1_mean, z1_std, hk.next_rng_key(), False)
        # p(z2(0) | z1(0))
        z2_mean, z2_std = z2_prior_init(z1)
        z2 = reparameterize_gaussian(z2_mean, z2_std, hk.next_rng_key(), False)

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        for t in range(1, action_.shape[1] + 1):
            # p(z1(t) | z2(t-1), a(t-1))
            z1_mean, z1_std = z1_prior(jnp.concatenate([z2_[t - 1], action_[:, t - 1]], axis=1))
            z1 = reparameterize_gaussian(z1_mean, z1_std, hk.next_rng_key(), False)
            # p(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = z2_prior(jnp.concatenate([z1, z2_[t - 1], action_[:, t - 1]], axis=1))
            z2 = reparameterize_gaussian(z2_mean, z2_std, hk.next_rng_key(), False)

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = jnp.stack(z1_mean_, axis=1)
        z1_std_ = jnp.stack(z1_std_, axis=1)
        z1_ = jnp.stack(z1_, axis=1)
        z2_ = jnp.stack(z2_, axis=1)

        return (z1_mean_, z1_std_, z1_, z2_)


class StochasticLatentVariablePosterior(hk.Module):
    """
    Posterior distribution of stochastic latent variable model for SLAC.
    """

    def __init__(
        self,
        action_space,
        z2_prior_init,
        z2_prior,
        feature_dim=256,
        z1_dim=32,
        z2_dim=256,
        hidden_units=(256, 256),
    ):
        super().__init__()
        self.action_space = action_space
        self.z2_prior_init = z2_prior_init
        self.z2_prior = z2_prior
        self.feature_dim = feature_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.hidden_units = hidden_units

    def __call__(self, feature_, action_):
        # q(z1(0) | feat(0))
        z1_posterior_init = Gaussian(self.feature_dim, self.z1_dim, self.hidden_units)
        # q(z2(0) | z1(0)) = p(z2(0) | z1(0))
        z2_posterior_init = self.z2_prior_init
        # q(z1(t+1) | feat(t+1), z2(t), a(t))
        z1_posterior = Gaussian(
            self.feature_dim + self.z2_dim + self.action_space.shape[0],
            self.z1_dim,
            self.hidden_units,
        )
        # q(z2(t+1) | z1(t+1), z2(t), a(t)) = p(z2(t+1) | z1(t+1), z2(t), a(t))
        z2_posterior = self.z2_prior

        z1_mean_ = []
        z1_std_ = []
        z1_ = []
        z2_ = []

        # p(z1(0)) = N(0, I)
        z1_mean, z1_std = z1_posterior_init(action_[:, 0])
        z1 = reparameterize_gaussian(z1_mean, z1_std, hk.next_rng_key(), False)
        # p(z2(0) | z1(0))
        z2_mean, z2_std = z2_posterior_init(z1)
        z2 = reparameterize_gaussian(z2_mean, z2_std, hk.next_rng_key(), False)

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        z1_mean_.append(z1_mean)
        z1_std_.append(z1_std)
        z1_.append(z1)
        z2_.append(z2)

        for t in range(1, action_.shape[1] + 1):
            # q(z1(t) | feat(t), z2(t-1), a(t-1))
            z1_mean, z1_std = z1_posterior(jnp.concatenate([feature_[:, t], z2_[t - 1], action_[:, t - 1]], axis=1))
            z1 = reparameterize_gaussian(z1_mean, z1_std, hk.next_rng_key(), False)
            # q(z2(t) | z1(t), z2(t-1), a(t-1))
            z2_mean, z2_std = z2_posterior(jnp.concatenate([z1, z2_[t - 1], action_[:, t - 1]], axis=1))
            z2 = reparameterize_gaussian(z2_mean, z2_std, hk.next_rng_key(), False)

            z1_mean_.append(z1_mean)
            z1_std_.append(z1_std)
            z1_.append(z1)
            z2_.append(z2)

        z1_mean_ = jnp.stack(z1_mean_, axis=1)
        z1_std_ = jnp.stack(z1_std_, axis=1)
        z1_ = jnp.stack(z1_, axis=1)
        z2_ = jnp.stack(z2_, axis=1)

        return (z1_mean_, z1_std_, z1_, z2_)
