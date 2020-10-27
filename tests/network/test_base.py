from functools import partial

import haiku as hk
import jax.numpy as jnp
import numpy as np
import pytest
from jax import nn

from rljax.network.base import MLP


@pytest.mark.parametrize(
    "input_dim, output_dim, hidden_units, hidden_activation, output_activation, d2rl",
    [
        (1, 1, (256,), nn.relu, None, False),
        (50, 1, (256,), nn.relu, None, False),
        (1, 50, (256,), nn.relu, None, False),
        (50, 50, (256, 256), nn.relu, None, True),
        (50, 50, (256,), nn.relu, nn.softmax, False),
        (50, 50, (256,), partial(nn.leaky_relu, negative_slope=0.2), jnp.tanh, False),
    ],
)
def test_mlp(input_dim, output_dim, hidden_units, hidden_activation, output_activation, d2rl):
    net = hk.without_apply_rng(
        hk.transform(
            lambda x: MLP(
                output_dim=output_dim,
                hidden_units=hidden_units,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                d2rl=d2rl,
            )(x)
        )
    )
    x = np.zeros((1, input_dim), dtype=np.float32)
    params = net.init(next(hk.PRNGSequence(0)), x)
    y = net.apply(params, x)
    assert y.shape == (1, output_dim)
