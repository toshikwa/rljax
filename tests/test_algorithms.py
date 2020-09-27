import os

import gym
import pytest

from rljax.algorithm import DDPG, DQN, PPO, QRDQN, SAC, TD3, SACDiscrete
from tensorboardX import SummaryWriter


def _test_algorithm(env, algo):
    state = env.reset()

    # Test step() method.
    for t in range(5):
        _state, _ = algo.step(env, state, t)
        assert env.observation_space.contains(_state)

    # Test select_action() method.
    action = algo.select_action(state)
    assert env.action_space.contains(action)

    # Test is_update() method.
    assert isinstance(algo.is_update(), bool)

    # Test update() method.
    algo.update(SummaryWriter(log_dir=os.path.join("/", "tmp", "rljax")))


@pytest.mark.parametrize(
    "use_per, dueling_net, double_q",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_qlearning(use_per, dueling_net, double_q):
    for ALGO in [DQN, QRDQN]:
        env = gym.make("CartPole-v0")
        algo = ALGO(
            num_steps=100000,
            state_space=env.observation_space,
            action_space=env.action_space,
            seed=0,
            use_per=use_per,
            dueling_net=dueling_net,
            double_q=double_q,
        )
        _test_algorithm(env, algo)


@pytest.mark.parametrize(
    "use_per, dueling_net",
    [
        (False, False),
        (True, False),
        (False, True),
    ],
)
def test_sac_discrete(use_per, dueling_net):
    env = gym.make("CartPole-v0")
    algo = SACDiscrete(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        use_per=use_per,
        dueling_net=dueling_net,
    )
    _test_algorithm(env, algo)


@pytest.mark.parametrize("use_per", [(False), (True)])
def test_actor_critic(use_per):
    for ALGO in [DDPG, TD3, SAC]:
        env = gym.make("MountainCarContinuous-v0")
        algo = ALGO(
            num_steps=100000,
            state_space=env.observation_space,
            action_space=env.action_space,
            seed=0,
            use_per=use_per,
        )
        _test_algorithm(env, algo)


def test_ppo():
    env = gym.make("MountainCarContinuous-v0")
    algo = PPO(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )
    _test_algorithm(env, algo)
