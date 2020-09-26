import jax.numpy as jnp
from flax import nn
from rljax.common.policy import StateIndependentGaussianPolicy
from rljax.common.value import ContinuousVFunction


def build_ppo_actor(state_dim, action_dim, rng_init, hidden_units=(64, 64)):
    """
    Build actor for PPO.
    """
    actor = StateIndependentGaussianPolicy.partial(
        action_dim=action_dim,
        hidden_units=hidden_units,
        hidden_activation=nn.tanh,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = actor.init_by_shape(rng_init, input_spec)
    return nn.Model(actor, param_init)


def build_ppo_critic(state_dim, rng_init, hidden_units=(64, 64)):
    """
    Build critic for PPO.
    """
    critic = ContinuousVFunction.partial(
        num_critics=1,
        hidden_units=hidden_units,
        hidden_activation=nn.tanh,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = critic.init_by_shape(rng_init, input_spec)
    return nn.Model(critic, param_init)
