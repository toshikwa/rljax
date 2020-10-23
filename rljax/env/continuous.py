import gym
import numpy as np
from gym.spaces import Box


def make_continuous_env(env_id):
    return NormalizedActionEnv(gym.make(env_id))


class NormalizedActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        # Original action space.
        self._low = env.action_space.low
        self._delta = env.action_space.high - env.action_space.low
        # Normalized action space.
        self.action_space = Box(low=-1.0, high=1.0, shape=env.action_space.shape, dtype=np.float64)

    def step(self, action):
        action = self._convert_action(action)
        return self.env.step(action)

    def _convert_action(self, action):
        return (action + 1.0) * self._delta / 2.0 + self._low
