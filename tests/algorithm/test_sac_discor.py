import gym
import pytest

from rljax.algorithm.sac_discor import SAC_DisCor

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
def test_sac_discor():
    env = gym.make("MountainCarContinuous-v0")
    algo = SAC_DisCor(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
    )
    _test_algorithm(env, algo)
