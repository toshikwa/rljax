from .ddpg import DDPG
from .dqn import DQN
from .ppo import PPO
from .qrdqn import QRDQN
from .sac import SAC
from .sac_discrete import SACDiscrete
from .td3 import TD3

CONTINUOUS_ALGORITHM = {
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
}
DISCRETE_ALGORITHM = {
    "dqn": DQN,
    "qrdqn": QRDQN,
    "sac_discrete": SACDiscrete,
}
PER_ALGORITHM = {
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "dqn": DQN,
    "qrdqn": QRDQN,
    "sac_discrete": SACDiscrete,
}
