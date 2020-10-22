import gym
import numpy as np

from rljax.buffer.rollout_buffer import RolloutBuffer


def test_rollout_buffer():
    for env_id in ["CartPole-v0", "MsPacmanNoFrameskip-v4", "Pendulum-v0"]:
        env = gym.make(env_id)
        state_dtype = np.float32
        state_shape = env.observation_space.shape
        action_dtype = np.float32 if env_id == "Pendulum-v0" else np.int32
        action_shape = env.action_space.shape if env_id == "Pendulum-v0" else (1,)

        buffer = RolloutBuffer(5, env.observation_space, env.action_space)
        state = np.stack([env.observation_space.sample() for _ in range(11)], axis=0).astype(state_dtype)
        action = np.stack([env.action_space.sample() for _ in range(10)], axis=0).astype(action_dtype)
        reward = np.random.rand(10, 1).astype(np.float32)
        done = np.random.rand(10, 1) < 0.5
        log_pi = np.random.rand(10, 1).astype(np.float32)

        for i in range(5):
            buffer.append(state[i], action[i], reward[i], done[i], log_pi[i], state[i + 1])
            assert (np.array(buffer.state[i]) == state[i]).all()
            assert (buffer.action[i] == action[i]).all()
            assert (buffer.reward[i] == reward[i]).all()
            assert (float(buffer.done[i]) == done[i]).all()
            assert (buffer.log_pi[i] == log_pi[i]).all()
            assert (np.array(buffer.next_state[i]) == state[i + 1]).all()

        for i in range(5, 10):
            buffer.append(state[i], action[i], reward[i], done[i], log_pi[i], state[i + 1])
            assert (np.array(buffer.state[i - 5]) == state[i]).all()
            assert (buffer.action[i - 5] == action[i]).all()
            assert (buffer.reward[i - 5] == reward[i]).all()
            assert (float(buffer.done[i - 5]) == done[i]).all()
            assert (buffer.log_pi[i - 5] == log_pi[i]).all()
            assert (np.array(buffer.next_state[i - 5]) == state[i + 1]).all()

        s, a, r, d, l, n_s = buffer.get()
        assert s.shape == (5,) + state_shape and s.dtype == state_dtype
        assert a.shape == (5,) + action_shape and a.dtype == action_dtype
        assert r.shape == (5, 1)
        assert d.shape == (5, 1)
        assert l.shape == (5, 1)
        assert n_s.shape == (5,) + state_shape and n_s.dtype == state_dtype
