import gym
import numpy as np
import pytest

from rljax.buffer.prioritized_buffer import PrioritizedReplayBuffer


@pytest.mark.parametrize(
    "env_id, state_dtype, state_shape, action_dtype, action_shape",
    [
        ("CartPole-v0", np.float32, (4,), np.int32, (1,)),
        ("MsPacmanNoFrameskip-v4", np.uint8, (210, 160, 3), np.int32, (1,)),
        ("Pendulum-v0", np.float32, (3,), np.float32, (1,)),
    ],
)
def test_prioritized_buffer(env_id, state_dtype, state_shape, action_dtype, action_shape):
    env = gym.make(env_id)

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
    state = np.stack([env.observation_space.sample() for _ in range(6)], axis=0).astype(state_dtype)
    action = np.stack([env.action_space.sample() for _ in range(5)], axis=0).astype(action_dtype)
    reward = np.random.rand(5, 1).astype(np.float32)
    done = np.random.rand(5, 1) < 0.5

    for i in range(5):
        buffer.append(state[i], action[i], reward[i, 0], done[i, 0], state[i + 1])
        assert np.isclose(np.array(buffer.state[i]), state[i]).all()
        assert np.isclose(buffer.action[i], action[i]).all()
        assert np.isclose(buffer.reward[i], reward[i]).all()
        assert np.isclose(float(buffer.done[i]), done[i]).all()
        assert np.isclose(np.array(buffer.next_state[i]), state[i + 1]).all()
        assert np.isclose(np.array(buffer.next_state[i]), state[i + 1]).all()
        assert np.isclose(buffer.tree_sum[i], 1.0)
        assert np.isclose(buffer.tree_min[i], 1.0)

    for i in range(3):
        w, (s, a, r, d, n_s) = buffer.sample(1)
        idxes = buffer._cached_idxes
        assert (0.0 <= w).all() and (w <= 1.0).all() and w.shape == (1, 1)
        assert s.shape == (1,) + state_shape and s.dtype == state_dtype
        assert a.shape == (1,) + action_shape and a.dtype == action_dtype
        assert r.shape == (1, 1)
        assert d.shape == (1, 1)
        assert n_s.shape == (1,) + state_shape and n_s.dtype == state_dtype
        assert np.isclose(buffer.beta, min(1.0, 0.4 + 0.6 / 4 * (i + 1)))

        abs_td = np.random.rand(1, 1).astype(np.float32)
        abs_td_target = np.clip((abs_td + 0.01) ** 0.6, 0.0, 1.0)
        buffer.update_priority(abs_td)

        for i, idx in enumerate(idxes):
            assert np.isclose(buffer.tree_sum[idx], abs_td_target[i, 0])
            assert np.isclose(buffer.tree_min[idx], abs_td_target[i, 0])


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
        assert np.isclose(abs_td_target, buffer._calculate_pa(abs_td)).all()
