import gym
import pytest

from rljax.algorithm.tqc import TQC

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
@pytest.mark.parametrize("d2rl", [(False), (True)])
def test_sac(d2rl):
    env = gym.make("MountainCarContinuous-v0")
    algo = TQC(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        d2rl=d2rl,
    )
    _test_algorithm(env, algo)
