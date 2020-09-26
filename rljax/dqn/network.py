import jax.numpy as jnp
from flax import nn
from rljax.common.value import DiscreteQFunction


def build_dqn(state_dim, action_dim, rng_init, hidden_units=(512,), dueling_net=True):
    """
    Build DQN.
    """
    dqn = DiscreteQFunction.partial(
        action_dim=action_dim,
        num_critics=1,
        hidden_units=hidden_units,
        hidden_activation=nn.relu,
        dueling_net=dueling_net,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = dqn.init_by_shape(rng_init, input_spec)
    return nn.Model(dqn, param_init)
