from .ddpg import DDPG
from .dqn import DQN
from .sac import SAC
from .td3 import TD3

CONTINUOUS_ALGOS = {
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
}
DISCRETE_ALGOS = {
    "dqn": DQN,
}
