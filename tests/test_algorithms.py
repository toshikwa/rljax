import gym
import numpy as np
import pytest

from rljax.algorithm import DDPG, DQN, FQF, IQN, PPO, QRDQN, SAC, TD3, DQN_DisCor, SAC_DisCor, SAC_Discrete


def _test_algorithm(env, algo):
    state = env.reset()

    # Test step() method.
    _state = algo.step(env, state)
    assert env.observation_space.contains(np.array(_state))

    # Test select_action() method.
    action = algo.select_action(state)
    assert env.action_space.contains(action)

    # Test is_update() method.
    assert isinstance(algo.is_update(), bool)

    # Test saving.
    algo.save_params("/tmp/rljax/test/algo")
    algo.load_params("/tmp/rljax/test/algo")


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
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        nstep=nstep,
        use_per=use_per,
        dueling_net=dueling_net,
        double_q=double_q,
    )
    _test_algorithm(env, algo)


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
    "use_per, dueling_net, double_q",
    [
        (False, False, False),
        (True, True, True),
    ],
)
def test_iqn(use_per, dueling_net, double_q):
    env = gym.make("CartPole-v0")
    algo = IQN(
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
    "use_per, dueling_net, double_q",
    [
        (False, False, False),
        (True, True, True),
    ],
)
def test_fqf(use_per, dueling_net, double_q):
    env = gym.make("CartPole-v0")
    algo = FQF(
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
        (True, False),
        (False, True),
    ],
)
def test_sac_discrete(use_per, dueling_net):
    env = gym.make("CartPole-v0")
    algo = SAC_Discrete(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        use_per=use_per,
        dueling_net=dueling_net,
    )
    _test_algorithm(env, algo)


def test_dqn_discor():
    env = gym.make("CartPole-v0")
    algo = DQN_DisCor(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
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


@pytest.mark.parametrize("use_per", [(False), (True)])
def test_ddpg(use_per):
    env = gym.make("MountainCarContinuous-v0")
    algo = DDPG(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        use_per=use_per,
    )
    _test_algorithm(env, algo)


@pytest.mark.parametrize("use_per", [(False), (True)])
def test_td3(use_per):
    env = gym.make("MountainCarContinuous-v0")
    algo = TD3(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        use_per=use_per,
    )
    _test_algorithm(env, algo)


@pytest.mark.parametrize("use_per", [(False), (True)])
def test_sac(use_per):
    env = gym.make("MountainCarContinuous-v0")
    algo = SAC(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        use_per=use_per,
    )
    _test_algorithm(env, algo)


def test_sac_discor():
    env = gym.make("MountainCarContinuous-v0")
    algo = SAC_DisCor(
        num_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )
    _test_algorithm(env, algo)
