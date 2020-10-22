import gym
import numpy as np
import pytest

from rljax.buffer.rollout_buffer import RolloutBuffer


@pytest.mark.parametrize(
    "env_id, state_dtype, state_shape, action_dtype, action_shape",
    [
        ("CartPole-v0", np.float32, (4,), np.int32, (1,)),
        ("MsPacmanNoFrameskip-v4", np.float32, (210, 160, 3), np.int32, (1,)),
        ("Pendulum-v0", np.float32, (3,), np.float32, (1,)),
    ],
)
def test_rollout_buffer(env_id, state_dtype, state_shape, action_dtype, action_shape):
    env = gym.make(env_id)
    buffer = RolloutBuffer(5, env.observation_space, env.action_space)
    state = np.stack([env.observation_space.sample() for _ in range(11)], axis=0).astype(state_dtype)
    action = np.stack([env.action_space.sample() for _ in range(10)], axis=0).astype(action_dtype)
    reward = np.random.rand(10, 1).astype(np.float32)
    done = np.random.rand(10, 1) < 0.5
    log_pi = np.random.rand(10, 1).astype(np.float32)

    for i in range(5):
        buffer.append(state[i], action[i], reward[i, 0], done[i, 0], log_pi[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state[i]), state[i]).all()
        assert np.isclose(buffer.action[i], action[i]).all()
        assert np.isclose(buffer.reward[i], reward[i]).all()
        assert np.isclose(float(buffer.done[i]), done[i]).all()
        assert np.isclose(buffer.log_pi[i], log_pi[i]).all()
        assert np.isclose(np.array(buffer.next_state[i]), state[i + 1]).all()

    for i in range(5, 10):
        buffer.append(state[i], action[i], reward[i, 0], done[i, 0], log_pi[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state[i - 5]), state[i]).all()
        assert np.isclose(buffer.action[i - 5], action[i]).all()
        assert np.isclose(buffer.reward[i - 5], reward[i]).all()
        assert np.isclose(float(buffer.done[i - 5]), done[i]).all()
        assert np.isclose(buffer.log_pi[i - 5], log_pi[i]).all()
        assert np.isclose(np.array(buffer.next_state[i - 5]), state[i + 1]).all()

    s, a, r, d, l, n_s = buffer.get()
    assert np.isclose(s, state[5:10]).all()
    assert np.isclose(a, action[5:10].reshape([5, 1])).all()
    assert np.isclose(r, reward[5:10]).all()
    assert np.isclose(d, done[5:10]).all()
    assert np.isclose(l, log_pi[5:10]).all()
    assert np.isclose(n_s, state[6:11]).all()
