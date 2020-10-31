import gym
import pytest

from rljax.algorithm.td3 import TD3

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
@pytest.mark.parametrize("use_per, d2rl", [(False, False), (True, True)])
def test_td3(use_per, d2rl):
    env = gym.make("MountainCarContinuous-v0")
    algo = TD3(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        use_per=use_per,
        d2rl=d2rl,
    )
    _test_algorithm(env, algo)
