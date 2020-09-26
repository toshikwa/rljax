import gym

from rljax.ddpg import DDPG
from rljax.dqn import DQN
from rljax.ppo import PPO
from rljax.sac import SAC
from rljax.sac_discrete import SACDiscrete
from rljax.td3 import TD3

gym.logger.set_level(40)

CONTINUOUS_ALGOS = {
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
}
DISCRETE_ALGOS = {
    "dqn": DQN,
    "sac_discrete": SACDiscrete,
}
