import jax.numpy as jnp
from flax import nn
from rljax.common.policy import StateDependentGaussianPolicy
from rljax.common.value import ContinuousQFunction


class LogAlpha(nn.Module):
    """
    Log of the entropy coefficient for SAC.
    """

    def apply(self):
        log_alpha = self.param("log_alpha", (), nn.initializers.zeros)
        return jnp.asarray(log_alpha, dtype=jnp.float32)


def build_sac_actor(state_dim, action_dim, rng_init, hidden_units=(256, 256)):
    """
    Build actor for SAC.
    """
    actor = StateDependentGaussianPolicy.partial(
        action_dim=action_dim,
        hidden_units=hidden_units,
        hidden_activation=nn.relu,
    )
    input_spec = [((1, state_dim), jnp.float32)]
    _, param_init = actor.init_by_shape(rng_init, input_spec)
    return nn.Model(actor, param_init)


def build_sac_critic(state_dim, action_dim, rng_init, hidden_units=(256, 256)):
    """
    Build critic for SAC.
    """
    critic = ContinuousQFunction.partial(
        q1=False,
        num_critics=2,
        hidden_units=hidden_units,
        hidden_activation=nn.relu,
    )
    input_spec = [
        ((1, state_dim), jnp.float32),
        ((1, action_dim), jnp.float32),
    ]
    _, param_init = critic.init_by_shape(rng_init, input_spec)
    return nn.Model(critic, param_init)


def build_sac_log_alpha(rng_init):
    """
    Build the model to optimize log(alpha) for SAC.
    """
    _, param_init = LogAlpha.init(rng_init)
    return nn.Model(LogAlpha, param_init)
