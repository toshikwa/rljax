import pytest

from rljax.algorithm.misc import SlacObservation
from rljax.algorithm.slac import SLAC


def _test_slac(env, algo):
    ob = SlacObservation(env.observation_space, env.action_space, 8)
    state = env.reset()
    ob.reset_episode(state)
    algo.buffer.reset_episode(state)

    for _ in range(10):
        # Test step() method.
        algo.step(env, ob)

        # Test select_action() method.
        action = algo.select_action(ob)
        assert env.action_space.contains(action)

    # Test is_update() method.
    assert isinstance(algo.is_update(), bool)

    # Test update_sac() method.
    algo.update_sac()
    # Test update_model() method.
    algo.update_model()
    # Test save_params() method.
    algo.save_params("/tmp/rljax/test")
    # Test load_params() method.
    algo.load_params("/tmp/rljax/test")


@pytest.mark.mujoco
@pytest.mark.slow
@pytest.mark.parametrize("d2rl", [(False), (True)])
def test_slac(d2rl):
    from rljax.env.mujoco.dmc import make_dmc_env

    env = make_dmc_env("cheetah", "run", 4, 1, 64)
    algo = SLAC(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size_model=2,
        batch_size_sac=2,
        start_steps=2,
        d2rl=d2rl,
    )
    _test_slac(env, algo)
