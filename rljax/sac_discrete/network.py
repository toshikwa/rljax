import jax.numpy as jnp
from flax import nn
from rljax.common.policy import CategoricalPolicy
from rljax.common.value import DiscreteQFunction


def build_sac_discrete_actor(state_dim, action_dim, rng_init, hidden_units=(512,)):
    """
    Build actor for SAC-Discrete.
    """
    actor = CategoricalPolicy.partial(
        action_dim=action_dim,
        hidden_units=hidden_units,
        hidden_activation=nn.relu,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = actor.init_by_shape(rng_init, input_spec)
    return nn.Model(actor, param_init)


def build_sac_discrete_critic(state_dim, action_dim, rng_init, hidden_units=(512,), dueling_net=True):
    """
    Build critic for SAC-Discrete.
    """
    critic = DiscreteQFunction.partial(
        action_dim=action_dim,
        num_critics=2,
        hidden_units=hidden_units,
        hidden_activation=nn.relu,
        dueling_net=dueling_net,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = critic.init_by_shape(rng_init, input_spec)
    return nn.Model(critic, param_init)
