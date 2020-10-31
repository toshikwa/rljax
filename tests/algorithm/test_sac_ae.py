import pytest

from rljax.algorithm.sac_ae import SAC_AE

from ._test_algorithm import _test_algorithm


@pytest.mark.mujoco
@pytest.mark.slow
@pytest.mark.parametrize("d2rl", [(False), (True)])
def test_sac_ae(d2rl):
    from rljax.env.mujoco.dmc import make_dmc_env

    env = make_dmc_env("cheetah", "run", 4)
    algo = SAC_AE(
        num_agent_steps=100000,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
        batch_size=4,
        start_steps=2,
        d2rl=d2rl,
    )
    _test_algorithm(env, algo)
