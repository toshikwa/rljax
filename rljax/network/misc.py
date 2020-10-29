from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import nn

from rljax.network.base import MLP
from rljax.network.conv import DQNBody, SLACDecoder, SLACEncoder


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


def make_quantile_nerwork(
    rng,
    state_space,
    action_space,
    fn,
    num_quantiles,
):
    """
    Make Quantile Nerwork for FQF.
    """
    fake_state = state_space.sample()[None, ...]
    if len(state_space.shape) == 1:
        fake_state = fake_state.astype(np.float32)
    network_dict = {}
    params_dict = {}

    if len(state_space.shape) == 3:
        network_dict["feature"] = hk.without_apply_rng(hk.transform(lambda s: DQNBody()(s)))
        fake_feature = np.zeros((1, 7 * 7 * 64), dtype=np.float32)
    else:
        network_dict["feature"] = hk.without_apply_rng(hk.transform(lambda s: s))
        fake_feature = fake_state
    params_dict["feature"] = network_dict["feature"].init(next(rng), fake_state)

    fake_cum_p = np.empty((1, num_quantiles), dtype=np.float32)
    network_dict["quantile"] = hk.without_apply_rng(hk.transform(fn))
    params_dict["quantile"] = network_dict["quantile"].init(next(rng), fake_feature, fake_cum_p)

    network_dict = hk.data_structures.to_immutable_dict(network_dict)
    params_dict = hk.data_structures.to_immutable_dict(params_dict)
    return network_dict, params_dict, fake_feature


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


def make_stochastic_latent_variable_model(
    rng,
    state_space,
    action_space,
    num_sequences,
    units_model,
    z1_dim,
    z2_dim,
    feature_dim,
):
    """
    Make Stochastic Latent Variable Model for SLAC.
    """

    # Fake input for JIT compilations.
    fake_state_ = jnp.empty((1, num_sequences, *state_space.shape), dtype=jnp.uint8)
    fake_action_ = jnp.empty((1, num_sequences, *action_space.shape))
    fake_action = jnp.empty((1, *action_space.shape))
    fake_feature = jnp.empty((1, feature_dim))
    fake_z_ = jnp.empty((1, num_sequences, z1_dim + z2_dim))
    fake_z1_ = jnp.empty((1, num_sequences, z1_dim))
    fake_z2_ = jnp.empty((1, num_sequences, z2_dim))
    fake_z1 = jnp.empty((1, z1_dim))
    fake_z2 = jnp.empty((1, z2_dim))

    def fn_z1_prior(z2, a):
        return Gaussian(output_dim=z1_dim, hidden_units=units_model)(jnp.concatenate([z2, a], axis=1))

    def fn_z1_post(f, z2, a):
        return Gaussian(output_dim=z1_dim, hidden_units=units_model)(jnp.concatenate([f, z2, a], axis=1))

    def fn_z2(z1, z2, a):
        return Gaussian(output_dim=z2_dim, hidden_units=units_model)(jnp.concatenate([z1, z2, a], axis=1))

    def fn_reward(z_, a_, n_z_):
        x = jnp.concatenate([z_, a_, n_z_], axis=-1)
        B, S, X = x.shape
        mean, std = Gaussian(output_dim=1, hidden_units=units_model)(x.reshape([B * S, X]))
        return mean.reshape([B, S, 1]), std.reshape([B, S, 1])

    def fn_encoder(x):
        return SLACEncoder(output_dim=feature_dim)(x)

    def fn_decoder(z1_, z2_):
        return SLACDecoder(state_space=state_space, std=np.sqrt(0.1))(jnp.concatenate([z1_, z2_], axis=-1))

    network_dict = {}
    params_dict = {}

    network_dict["z1_prior_init"] = hk.without_apply_rng(hk.transform(lambda x: ConstantGaussian(z1_dim, 1.0)(x)))
    params_dict["z1_prior_init"] = network_dict["z1_prior_init"].init(next(rng), fake_action)

    network_dict["z1_prior"] = hk.without_apply_rng(hk.transform(fn_z1_prior))
    params_dict["z1_prior"] = network_dict["z1_prior"].init(next(rng), fake_z2, fake_action)

    network_dict["z1_post_init"] = hk.without_apply_rng(hk.transform(lambda x: Gaussian(z1_dim, units_model)(x)))
    params_dict["z1_post_init"] = network_dict["z1_post_init"].init(next(rng), fake_feature)

    network_dict["z1_post"] = hk.without_apply_rng(hk.transform(fn_z1_post))
    params_dict["z1_post"] = network_dict["z1_post"].init(next(rng), fake_feature, fake_z2, fake_action)

    network_dict["z2_init"] = hk.without_apply_rng(hk.transform(lambda x: Gaussian(z2_dim, units_model)(x)))
    params_dict["z2_init"] = network_dict["z2_init"].init(next(rng), fake_z1)

    network_dict["z2"] = hk.without_apply_rng(hk.transform(fn_z2))
    params_dict["z2"] = network_dict["z2"].init(next(rng), fake_z1, fake_z2, fake_action)

    network_dict["reward"] = hk.without_apply_rng(hk.transform(fn_reward))
    params_dict["reward"] = network_dict["reward"].init(next(rng), fake_z_, fake_action_, fake_z_)

    network_dict["encoder"] = hk.without_apply_rng(hk.transform(fn_encoder))
    params_dict["encoder"] = network_dict["encoder"].init(next(rng), fake_state_)

    network_dict["decoder"] = hk.without_apply_rng(hk.transform(fn_decoder))
    params_dict["decoder"] = network_dict["decoder"].init(next(rng), fake_z1_, fake_z2_)

    network_dict = hk.data_structures.to_immutable_dict(network_dict)
    params_dict = hk.data_structures.to_immutable_dict(params_dict)
    return network_dict, params_dict
