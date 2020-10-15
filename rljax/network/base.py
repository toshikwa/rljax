import haiku as hk
import jax.numpy as jnp
import numpy as np
from jax import nn


class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation=nn.relu,
        output_activation=None,
        output_scale=1.0,
    ):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.output_kwargs = {}
        if output_scale != 1.0:
            self.output_kwargs["w_init"] = hk.initializers.VarianceScaling(scale=output_scale)

    def __call__(self, x):
        for unit in self.hidden_units:
            x = hk.Linear(unit)(x)
            x = self.hidden_activation(x)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class DQNBody(hk.Module):
    """
    CNN for the atari environment.
    """

    def __init__(self):
        super(DQNBody, self).__init__()

    def __call__(self, x):
        # He's initializer.
        w_init = hk.initializers.VarianceScaling(scale=np.square(2), distribution="uniform")
        # Floatify the image.
        x = x.astype(jnp.float32) / 255.0
        # Apply CNN.
        x = hk.Conv2D(32, kernel_shape=8, stride=4, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=4, stride=2, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        x = hk.Conv2D(64, kernel_shape=3, stride=1, padding="VALID", w_init=w_init)(x)
        x = nn.relu(x)
        # Flatten the feature map.
        return hk.Flatten()(x)
