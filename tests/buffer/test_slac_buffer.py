import gym
import numpy as np
import pytest

from rljax.buffer.slac_buffer import SLACReplayBuffer


@pytest.mark.parametrize(
    "env_id, state_dtype, state_shape, action_dtype, action_shape",
    [
        ("CartPole-v0", np.float32, (4,), np.int32, (1,)),
        ("MsPacmanNoFrameskip-v4", np.uint8, (210, 160, 3), np.int32, (1,)),
        ("Pendulum-v0", np.float32, (3,), np.float32, (1,)),
    ],
)
def test_slac_buffer(env_id, state_dtype, state_shape, action_dtype, action_shape):
    env = gym.make(env_id)
    buffer = SLACReplayBuffer(10, env.observation_space, env.action_space, num_sequences=4)
    state = np.stack([env.observation_space.sample() for _ in range(11)], axis=0).astype(state_dtype)
    action = np.stack([env.action_space.sample() for _ in range(10)], axis=0).astype(action_dtype)
    reward = np.random.rand(10, 1).astype(np.float32)
    done = np.random.rand(10, 1) < 0.5

    buffer.reset_episode(state[0])
    for i in range(3):
        buffer.append(action[i], reward[i, 0], done[i, 0], state[i + 1])

    for i in range(3, 10):
        buffer.append(action[i], reward[i, 0], done[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state_[i - 3]), state[i - 3 : i + 2]).all()
        assert np.isclose(buffer.action_[i - 3], action[i - 3 : i + 1].reshape([4, 1])).all()
        assert np.isclose(buffer.reward_[i - 3], reward[i - 3 : i + 1]).all()
        assert np.isclose(buffer.done_[i - 3], done[i - 3 : i + 1]).all()

    s_, a_, r_, d_ = buffer.sample_model(3)
    assert s_.shape == (3, 5) + state_shape and s_.dtype == state_dtype
    assert a_.shape == (3, 4) + action_shape and a_.dtype == action_dtype
    assert r_.shape == (3, 4, 1)
    assert d_.shape == (3, 4, 1)

    s_, a_, r, d = buffer.sample_sac(3)
    assert s_.shape == (3, 5) + state_shape and s_.dtype == state_dtype
    assert a_.shape == (3, 4) + action_shape and a_.dtype == action_dtype
    assert r.shape == (3, 1)
    assert d.shape == (3, 1)
