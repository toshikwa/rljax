import gym
import numpy as np

from rljax.buffer.replay_buffer import NStepBuffer, ReplayBuffer


def test_nstep_buffer():
    buffer = NStepBuffer(gamma=0.99, nstep=3)
    state = np.random.rand(5, 1)
    action = np.random.rand(5, 1)
    reward = np.random.rand(5, 1)

    for i in range(2):
        buffer.append(state[i], action[i], reward[i])
        assert not buffer.is_full() and not buffer.is_empty()

    for i in range(2, 5):
        buffer.append(state[i], action[i], reward[i])
        assert buffer.is_full() and not buffer.is_empty()

        s, a, r = buffer.get()
        assert not buffer.is_full() and not buffer.is_empty()
        assert r == reward[i - 2, 0] + reward[i - 1, 0] * 0.99 + reward[i, 0] * (0.99 ** 2)

    s, a, r = buffer.get()
    assert not buffer.is_full() and not buffer.is_empty()
    assert r == reward[-2, 0] + reward[-1, 0] * 0.99

    s, a, r = buffer.get()
    assert not buffer.is_full() and buffer.is_empty()
    assert r == reward[-1, 0]


def test_replay_buffer():
    for env_id in ["CartPole-v0", "MsPacmanNoFrameskip-v4", "Pendulum-v0"]:
        env = gym.make(env_id)
        state_dtype = np.uint8 if env_id == "MsPacmanNoFrameskip-v4" else np.float32
        state_shape = env.observation_space.shape
        action_dtype = np.float32 if env_id == "Pendulum-v0" else np.int32
        action_shape = env.action_space.shape if env_id == "Pendulum-v0" else (1,)

        buffer = ReplayBuffer(5, env.observation_space, env.action_space, gamma=0.99, nstep=1)
        state = np.stack([env.observation_space.sample() for _ in range(11)], axis=0).astype(state_dtype)
        action = np.stack([env.action_space.sample() for _ in range(10)], axis=0).astype(action_dtype)
        reward = np.random.rand(10, 1).astype(np.float32)
        done = np.random.rand(10, 1) < 0.5

        for i in range(5):
            buffer.append(state[i], action[i], reward[i], done[i], state[i + 1])
            assert (np.array(buffer.state[i]) == state[i]).all()
            assert (buffer.action[i] == action[i]).all()
            assert (buffer.reward[i] == reward[i]).all()
            assert (float(buffer.done[i]) == done[i]).all()
            assert (np.array(buffer.next_state[i]) == state[i + 1]).all()

        for i in range(5, 10):
            buffer.append(state[i], action[i], reward[i], done[i], state[i + 1])
            assert (np.array(buffer.state[i - 5]) == state[i]).all()
            assert (buffer.action[i - 5] == action[i]).all()
            assert (buffer.reward[i - 5] == reward[i]).all()
            assert (float(buffer.done[i - 5]) == done[i]).all()
            assert (np.array(buffer.next_state[i - 5]) == state[i + 1]).all()

        weight, (state, action, reward, done, next_state) = buffer.sample(3)
        assert weight == 1.0
        assert state.shape == (3,) + state_shape and state.dtype == state_dtype
        assert action.shape == (3,) + action_shape and action.dtype == action_dtype
        assert reward.shape == (3, 1)
        assert done.shape == (3, 1)
        assert next_state.shape == (3,) + state_shape and next_state.dtype == state_dtype
