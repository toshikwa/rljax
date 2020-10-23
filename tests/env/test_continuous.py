import gym
import numpy as np
import pytest

from rljax.env.continuous import NormalizedActionEnv


def make_continuous_env(env_id):
    return NormalizedActionEnv(gym.make(env_id))


@pytest.mark.parametrize("env_id", [("Pendulum-v0")])
def test_normalized_action_env(env_id):
    env = gym.make(env_id)
    env_norm = NormalizedActionEnv(env)
    assert np.isclose(env_norm._convert_action(1.0), env.action_space.high[0])
    assert np.isclose(env_norm._convert_action(-1.0), env.action_space.low[0])
