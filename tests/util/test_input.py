import gym
import numpy as np

from rljax.env import make_atari_env
from rljax.util.input import fake_action, fake_state


def test_pixel_state():
    env = make_atari_env("MsPacmanNoFrameskip-v4")
    state = fake_state(env.observation_space)
    assert state.shape == (1, 84, 84, 4)
    assert state.dtype == np.uint8


def test_vector_state():
    env = gym.make("CartPole-v0")
    state = fake_state(env.observation_space)
    assert state.shape == (1, 4)
    assert state.dtype == np.float32


def test_fake_action():
    env = gym.make("Pendulum-v0")
    action = fake_action(env.action_space)
    assert action.shape == (1, 1)
    assert action.dtype == np.float32
