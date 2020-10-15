import random

import gym
from gym.envs.registration import register

import metaworld
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS

gym.logger.set_level(40)


def assert_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    assert hasattr(env, "_max_episode_steps")


METAWORLD_ENV_IDS = ("hammer-v1", "stick-push-v1", "push-wall-v1", "stick-pull-v1", "dial-turn-v1", "peg-insert-side-v1")
for env_id in METAWORLD_ENV_IDS:
    register(id=env_id, entry_point=ALL_V1_ENVIRONMENTS[env_id], max_episode_steps=150)
    assert_env(gym.make(env_id))


def make_metaworld_env(env_id, seed):
    assert env_id in METAWORLD_ENV_IDS
    env = gym.make(env_id)
    tasks = metaworld.MT1(env_id).train_tasks
    env.set_task(tasks[seed % len(tasks)])
    return env
