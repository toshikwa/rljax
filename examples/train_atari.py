import argparse
import os
from datetime import datetime

from rljax.algorithm import DISCRETE_ALGORITHM
from rljax.env import make_atari_env
from rljax.trainer import Trainer

config = {
    "dqn": {
        "nstep": 1,
        "loss_type": "l2",
        "dueling_net": True,
        "double_q": True,
    },
    "qrdqn": {
        "nstep": 1,
        "loss_type": "l2",
        "dueling_net": True,
        "double_q": True,
    },
    "iqn": {
        "nstep": 1,
        "loss_type": "l2",
        "dueling_net": True,
        "double_q": True,
    },
    "dqn_discor": {
        "nstep": 1,
        "loss_type": "l2",
        "dueling_net": True,
        "double_q": True,
    },
    "sac_discrete": {
        "nstep": 1,
        "dueling_net": True,
    },
}


def run(args):
    env = make_atari_env(args.env_id)
    env_test = make_atari_env(args.env_id, episode_life=False, clip_rewards=False)

    algo = DISCRETE_ALGORITHM[args.algo](
        num_steps=args.num_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="dqn")
    p.add_argument("--num_steps", type=int, default=5 * 10 ** 6)
    p.add_argument("--eval_interval", type=int, default=20000)
    p.add_argument("--env_id", type=str, default="MsPacmanNoFrameskip-v4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
