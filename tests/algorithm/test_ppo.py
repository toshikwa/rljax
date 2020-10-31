import gym
import pytest

from rljax.algorithm.ppo import PPO

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
def test_ppo():
    env = gym.make("MountainCarContinuous-v0")
    algo = PPO(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        buffer_size=4,
    )
    _test_algorithm(env, algo)
