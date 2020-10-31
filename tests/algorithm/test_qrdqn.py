import gym
import pytest

from rljax.algorithm.qrdqn import QRDQN

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
@pytest.mark.parametrize(
    "use_per, dueling_net, double_q",
    [
        (False, False, False),
        (True, True, True),
    ],
)
def test_qrdqn(use_per, dueling_net, double_q):
    env = gym.make("CartPole-v0")
    algo = QRDQN(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        use_per=use_per,
        dueling_net=dueling_net,
        double_q=double_q,
    )
    _test_algorithm(env, algo)
