import gym

from rljax import CONTINUOUS_ALGOS, DISCRETE_ALGOS


def _test_algorithm(env, state, ALGO):
    algo = ALGO(
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )

    # Test step() method.
    _state, _ = algo.step(env, state, 0, 0)
    assert env.observation_space.contains(_state)
    _state, _ = algo.step(env, state, 1000000, 1000000)
    assert env.observation_space.contains(_state)

    # Test select_action() method.
    action = algo.select_action(state)
    assert env.action_space.contains(action)

    # Test is_update() method.
    assert isinstance(algo.is_update(0), bool)
    assert isinstance(algo.is_update(1000000), bool)

    # Test update() method.
    algo.update()


def test_continuous_algorithms():
    env = gym.make("MountainCarContinuous-v0")
    state = env.reset()

    for ALGO in CONTINUOUS_ALGOS.values():
        _test_algorithm(env, state, ALGO)


def test_discrete_algorithms():
    env = gym.make("CartPole-v0")
    state = env.reset()

    for ALGO in DISCRETE_ALGOS.values():
        _test_algorithm(env, state, ALGO)
