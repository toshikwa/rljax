import gym
import pytest

from rljax.algorithm.dqn import DQN

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
@pytest.mark.parametrize(
    "nstep, use_per, dueling_net, double_q",
    [
        (1, False, False, False),
        (3, True, True, True),
    ],
)
def test_dqn(nstep, use_per, dueling_net, double_q):
    env = gym.make("CartPole-v0")
    algo = DQN(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        nstep=nstep,
        use_per=use_per,
        dueling_net=dueling_net,
        double_q=double_q,
    )
    _test_algorithm(env, algo)
