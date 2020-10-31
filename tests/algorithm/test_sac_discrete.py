import gym
import pytest

from rljax.algorithm.sac_discrete import SAC_Discrete

from ._test_algorithm import _test_algorithm


@pytest.mark.slow
@pytest.mark.parametrize(
    "use_per, dueling_net",
    [
        (True, False),
        (False, True),
    ],
)
def test_sac_discrete(use_per, dueling_net):
    env = gym.make("CartPole-v0")
    algo = SAC_Discrete(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        use_per=use_per,
        dueling_net=dueling_net,
    )
    _test_algorithm(env, algo)
