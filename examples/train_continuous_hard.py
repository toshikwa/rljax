import argparse
import os
from datetime import datetime

import gym

from rljax.algorithm import CONTINUOUS_ALGORITHM
from rljax.trainer import Trainer

config = {
    "ppo": {
        "buffer_size": 2048,
        "batch_size": 64,
        "epoch_ppo": 10,
    },
    "ddpg": {
        "use_per": False,
        "start_steps": 10000,
    },
    "td3": {
        "use_per": False,
        "start_steps": 10000,
    },
    "sac": {
        "use_per": False,
        "start_steps": 10000,
    },
}


def run(args):
    env = gym.make(args.env_id)
    env_test = gym.make(args.env_id)

    algo = CONTINUOUS_ALGORITHM[args.algo](
        num_steps=args.num_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        **config[args.algo],
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
    p.add_argument("--algo", type=str, default="sac")
    p.add_argument("--num_steps", type=int, default=3000000)
    p.add_argument("--eval_interval", type=int, default=20000)
    p.add_argument("--env_id", type=str, default="HalfCheetah-v3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
