from .ddpg import DDPG
from .sac import SAC

CONTINUOUS_ALGOS = {
    "ddpg": DDPG,
    "sac": SAC,
}
