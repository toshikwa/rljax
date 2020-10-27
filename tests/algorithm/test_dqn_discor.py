import gym
import pytest

from rljax.algorithm.dqn_discor import DQN_DisCor

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
def test_dqn_discor():
    env = gym.make("CartPole-v0")
    algo = DQN_DisCor(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )
    _test_algorithm(env, algo)
