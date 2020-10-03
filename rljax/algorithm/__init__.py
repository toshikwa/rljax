from .ddpg import DDPG
from .dqn import DQN
from .dqn_discor import DQN_DisCor
from .iqn import IQN
from .ppo import PPO
from .qrdqn import QRDQN
from .sac import SAC
from .sac_discor import SAC_DisCor
from .sac_discrete import SAC_Discrete
from .td3 import TD3

CONTINUOUS_ALGORITHM = {
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "sac_discor": SAC_DisCor,
}
DISCRETE_ALGORITHM = {
    "dqn": DQN,
    "iqn": IQN,
    "qrdqn": QRDQN,
    "sac_discrete": SAC_Discrete,
    "dqn_discor": DQN_DisCor,
}
