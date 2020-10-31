import numpy as np


def _test_algorithm(env, algo):
    state = env.reset()

    for _ in range(4):
        # Test step() method.
        _state = algo.step(env, state)
        assert env.observation_space.contains(np.array(_state))

        # Test select_action() method.
        action = algo.select_action(state)
        assert env.action_space.contains(action)

        state = _state

    # Test is_update() method.
    assert isinstance(algo.is_update(), bool)

    # Test update() method.
    algo.update()
    # Test save_params() method.
    algo.save_params("/tmp/rljax/test")
    # Test load_params() method.
    algo.load_params("/tmp/rljax/test")
