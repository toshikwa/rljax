import gym
import numpy as np
import pytest

from rljax.buffer.replay_buffer import NStepBuffer, ReplayBuffer


def test_nstep_buffer():
    buffer = NStepBuffer(gamma=0.99, nstep=3)
    state = np.random.rand(5, 1).astype(np.float32)
    action = np.random.rand(5, 1).astype(np.float32)
    reward = np.random.rand(5, 1).astype(np.float32)

    for i in range(2):
        buffer.append(state[i], action[i], reward[i, 0])
        assert not buffer.is_full() and not buffer.is_empty()

    for i in range(2, 5):
        buffer.append(state[i], action[i], reward[i, 0])
        assert buffer.is_full() and not buffer.is_empty()
        s, a, r = buffer.get()
        assert not buffer.is_full() and not buffer.is_empty()
        assert np.isclose(r, reward[i - 2, 0] + reward[i - 1, 0] * 0.99 + reward[i, 0] * (0.99 ** 2))

    s, a, r = buffer.get()
    assert not buffer.is_full() and not buffer.is_empty()
    assert np.isclose(r, reward[-2, 0] + reward[-1, 0] * 0.99)

    s, a, r = buffer.get()
    assert not buffer.is_full() and buffer.is_empty()
    assert np.isclose(r, reward[-1, 0])


@pytest.mark.parametrize(
    "env_id, state_dtype, state_shape, action_dtype, action_shape",
    [
        ("CartPole-v0", np.float32, (4,), np.int32, (1,)),
        ("MsPacmanNoFrameskip-v4", np.uint8, (210, 160, 3), np.int32, (1,)),
        ("Pendulum-v0", np.float32, (3,), np.float32, (1,)),
    ],
)
def test_replay_buffer(env_id, state_dtype, state_shape, action_dtype, action_shape):
    env = gym.make(env_id)
    buffer = ReplayBuffer(5, env.observation_space, env.action_space, gamma=0.99, nstep=1)
    state = np.stack([env.observation_space.sample() for _ in range(11)], axis=0).astype(state_dtype)
    action = np.stack([env.action_space.sample() for _ in range(10)], axis=0).astype(action_dtype)
    reward = np.random.rand(10, 1).astype(np.float32)
    done = np.random.rand(10, 1) < 0.5

    for i in range(5):
        buffer.append(state[i], action[i], reward[i, 0], done[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state[i]), state[i]).all()
        assert np.isclose(buffer.action[i], action[i]).all()
        assert np.isclose(buffer.reward[i], reward[i]).all()
        assert np.isclose(float(buffer.done[i]), done[i]).all()
        assert np.isclose(np.array(buffer.next_state[i]), state[i + 1]).all()

    for i in range(5, 10):
        buffer.append(state[i], action[i], reward[i, 0], done[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state[i - 5]), state[i]).all()
        assert np.isclose(buffer.action[i - 5], action[i]).all()
        assert np.isclose(buffer.reward[i - 5], reward[i]).all()
        assert np.isclose(float(buffer.done[i - 5]), done[i]).all()
        assert np.isclose(np.array(buffer.next_state[i - 5]), state[i + 1]).all()

    w, (s, a, r, d, n_s) = buffer.sample(3)
    assert w == 1.0
    assert s.shape == (3,) + state_shape and s.dtype == state_dtype
    assert a.shape == (3,) + action_shape and a.dtype == action_dtype
    assert r.shape == (3, 1)
    assert d.shape == (3, 1)
    assert n_s.shape == (3,) + state_shape and n_s.dtype == state_dtype
