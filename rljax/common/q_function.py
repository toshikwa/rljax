import jax.numpy as jnp
from flax import nn


class ContinuousQFunction(nn.Module):
    """
    Critic for DDPG, TD3 and SAC.
    """

    def apply(self, state, action, q1=False, num_critics=2, hidden_units=(400, 300), hidden_activation=nn.relu):
        x = jnp.concatenate([state, action], axis=1)

        def q_func(x):
            for unit in hidden_units:
                x = nn.Dense(x, features=unit)
                x = nn.relu(x)
            return nn.Dense(x, features=1)

        x1 = q_func(x)
        if q1:
            return x1

        xs = [x1] + [q_func(x) for _ in range(num_critics - 1)]
        return xs


class DiscreteQFunction(nn.Module):
    """
    Critic for DQN.
    """

    def apply(self, x, action_dim, hidden_units=(512,), hidden_activation=nn.relu, dueling_net=False):
        a = x
        for unit in hidden_units:
            a = nn.Dense(a, features=unit)
            a = nn.relu(a)
        a = nn.Dense(a, features=action_dim)
        if not dueling_net:
            return a

        v = x
        for unit in hidden_units:
            v = nn.Dense(v, features=unit)
            v = nn.relu(v)
        v = nn.Dense(v, features=1)
        return a + v - a.mean(axis=1, keepdims=True)
