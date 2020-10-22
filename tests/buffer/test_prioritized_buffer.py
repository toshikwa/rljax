import gym
import numpy as np

from rljax.buffer.prioritized_buffer import PrioritizedReplayBuffer


def test_prioritized_buffer():
    env = gym.make("CartPole-v0")
    state_shape = env.observation_space.shape
    action_shape = (1,)

    buffer = PrioritizedReplayBuffer(
        5,
        env.observation_space,
        env.action_space,
        gamma=0.99,
        nstep=1,
        alpha=0.6,
        beta=0.4,
        beta_steps=4,
        min_pa=0.0,
        max_pa=1.0,
        eps=0.01,
    )
    state = np.stack([env.observation_space.sample() for _ in range(6)], axis=0).astype(np.float32)
    action = np.stack([env.action_space.sample() for _ in range(5)], axis=0).astype(np.int32)
    reward = np.random.rand(5, 1).astype(np.float32)
    done = np.random.rand(5, 1) < 0.5

    for i in range(5):
        buffer.append(state[i], action[i], reward[i], done[i], state[i + 1])
        assert (np.array(buffer.state[i]) == state[i]).all()
        assert (buffer.action[i] == action[i]).all()
        assert (buffer.reward[i] == reward[i]).all()
        assert (float(buffer.done[i]) == done[i]).all()
        assert (np.array(buffer.next_state[i]) == state[i + 1]).all()
        assert np.array(buffer.next_state[i] == state[i + 1]).all()
        assert buffer.tree_sum[i] == 1.0
        assert buffer.tree_min[i] == 1.0

    for i in range(3):
        w, (s, a, r, d, n_s) = buffer.sample(1)
        idxes = buffer._cached_idxes
        assert (0.0 <= w).all() and (w <= 1.0).all() and w.shape == (1, 1)
        assert s.shape == (1,) + state_shape and s.dtype == np.float32
        assert a.shape == (1,) + action_shape and a.dtype == np.int32
        assert r.shape == (1, 1)
        assert d.shape == (1, 1)
        assert n_s.shape == (1,) + state_shape and n_s.dtype == np.float32
        assert round(buffer.beta, 8) == round(min(1.0, 0.4 + 0.6 / 4 * (i + 1)), 8)

        abs_td = np.random.rand(1, 1).astype(np.float32)
        abs_td_target = np.clip((abs_td + 0.01) ** 0.6, 0.0, 1.0)
        buffer.update_priority(abs_td)

        for i, idx in enumerate(idxes):
            assert buffer.tree_sum[idx] == abs_td_target[i, 0]
            assert buffer.tree_min[idx] == abs_td_target[i, 0]


def test_calculate_pa():
    env = gym.make("CartPole-v0")
    buffer = PrioritizedReplayBuffer(
        1,
        env.observation_space,
        env.action_space,
        gamma=0.99,
        nstep=1,
        alpha=0.6,
        beta=0.4,
        beta_steps=4,
        min_pa=0.0,
        max_pa=1.0,
        eps=0.01,
    )

    for _ in range(3):
        abs_td = np.random.rand(3, 1).astype(np.float32)
        abs_td_target = np.clip((abs_td + 0.01) ** 0.6, 0.0, 1.0)
        assert (abs_td_target == buffer._calculate_pa(abs_td)).all()
