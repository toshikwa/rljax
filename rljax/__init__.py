import gym

from rljax.algorithm import DDPG, DQN, PPO, SAC, TD3, SACDiscrete

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
