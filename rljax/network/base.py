import haiku as hk
import jax.numpy as jnp
from jax import nn


class MLP(hk.Module):
    def __init__(
        self,
        output_dim,
        hidden_units,
        hidden_activation=nn.relu,
        output_activation=None,
        hidden_scale=1.0,
        output_scale=1.0,
        d2rl=False,
    ):
        super(MLP, self).__init__()
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.d2rl = d2rl
        self.hidden_kwargs = {"w_init": hk.initializers.Orthogonal(scale=hidden_scale)}
        self.output_kwargs = {"w_init": hk.initializers.Orthogonal(scale=output_scale)}

    def __call__(self, x):
        x_input = x
        for i, unit in enumerate(self.hidden_units):
            x = hk.Linear(unit, **self.hidden_kwargs)(x)
            x = self.hidden_activation(x)
            if self.d2rl and i + 1 != len(self.hidden_units):
                x = jnp.concatenate([x, x_input], axis=1)
        x = hk.Linear(self.output_dim, **self.output_kwargs)(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
