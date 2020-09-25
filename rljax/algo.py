from .ddpg import DDPG
from .sac import SAC
from .td3 import TD3

CONTINUOUS_ALGOS = {
    'ddpg': DDPG,
    'td3': TD3,
    'sac': SAC,
}
