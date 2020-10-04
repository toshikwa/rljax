import dmc2gym
import gym
import numpy as np

from .atari import FrameStack

gym.logger.set_level(40)


def make_dmc_env(domain_name, task_name, action_repeat, n_frames=3, image_size=84):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat,
    )
    env = PyTorch2Jax(env)
    env = FrameStack(env, n_frames=n_frames)
    setattr(env, "_max_episode_steps", env.env._max_episode_steps)
    return env


class PyTorch2Jax(gym.Wrapper):
    """
    Convert (C, H, W) into (H, W, C).
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[1], shp[2], shp[0]),
            dtype=env.observation_space.dtype,
        )

    def _get_ob(self, ob):
        return np.transpose(ob, (1, 2, 0)).copy()

    def reset(self):
        return self._get_ob(self.env.reset())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        return self._get_ob(ob), reward, done, info
