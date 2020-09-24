import jax
import jax.numpy as jnp
from flax import nn
from rljax.common.utils import calculate_log_pi


class SACActor(nn.Module):
    """
    Actor for Soft Actor-Critic(SAC).
    """

    def apply(self, x, action_dim, key=None, hidden_units=(256, 256), deterministic=True):
        for unit in hidden_units:
            x = nn.Dense(x, features=unit)
            x = nn.relu(x)
        x = nn.Dense(x, features=2 * action_dim)
        mean, log_std = jnp.split(x, 2, axis=1)

        if deterministic:
            return nn.tanh(mean)

        else:
            log_std = jnp.clip(log_std, -20, 2)
            std = jnp.exp(log_std)

            noise = jax.random.normal(key, std.shape)
            action = jnp.tanh(mean + noise * std)
            log_pi = calculate_log_pi(log_std, noise, action)
            return action, log_pi


class SACCritic(nn.Module):
    """
    Critic for Soft Actor-Critic(SAC).
    """

    def apply(self, state, action, key=None, hidden_units=(256, 256)):
        x1 = jnp.concatenate([state, action], axis=1)
        for unit in hidden_units:
            x1 = nn.Dense(x1, features=unit)
            x1 = nn.relu(x1)
        x1 = nn.Dense(x1, features=1)

        x2 = jnp.concatenate([state, action], axis=1)
        for unit in hidden_units:
            x2 = nn.Dense(x2, features=unit)
            x2 = nn.relu(x2)
        x2 = nn.Dense(x2, features=1)

        return x1, x2


class LogAlpha(nn.Module):
    """
    Log of the entropy coefficient for Soft Actor-Critic(SAC).
    """

    def apply(self):
        log_alpha = self.param("log_alpha", (), nn.initializers.zeros)
        return jnp.asarray(log_alpha, dtype=jnp.float32)


def build_sac_actor(state_shape, action_shape, rng_init, hidden_units=(256, 256)):
    actor = SACActor.partial(action_dim=action_shape[0], hidden_units=hidden_units)
    input_spec = [((1, state_shape[0]), jnp.float32)]
    _, param_init = actor.init_by_shape(rng_init, input_spec)
    return nn.Model(actor, param_init)


def build_sac_critic(state_shape, action_shape, rng_init, hidden_units=(256, 256)):
    critic = SACCritic.partial(hidden_units=hidden_units)
    input_spec = [
        ((1, state_shape[0]), jnp.float32),
        ((1, action_shape[0]), jnp.float32),
    ]
    _, param_init = critic.init_by_shape(rng_init, input_spec)
    return nn.Model(critic, param_init)


def build_sac_log_alpha(rng_init):
    _, param_init = LogAlpha.init(rng_init)
    return nn.Model(LogAlpha, param_init)
