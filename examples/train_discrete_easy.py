import argparse
import os
from datetime import datetime

import gym

from rljax.algorithm import DISCRETE_ALGORITHM
from rljax.trainer import Trainer

config = {
    "dqn": {
        "batch_size": 256,
        "start_steps": 1000,
        "update_interval": 1,
        "update_interval_target": 400,
        "loss_type": "l2",
        "lr": 1e-3,
    },
    "qrdqn": {
        "batch_size": 256,
        "start_steps": 1000,
        "update_interval": 1,
        "update_interval_target": 400,
        "num_quantiles": 21,
        "loss_type": "l2",
        "lr": 1e-3,
    },
    "iqn": {
        "batch_size": 256,
        "start_steps": 1000,
        "update_interval": 1,
        "update_interval_target": 400,
        "num_quantiles": 8,
        "num_cosines": 8,
        "loss_type": "l2",
        "lr": 1e-3,
    },
    "sac_discrete": {
        "batch_size": 256,
        "start_steps": 1000,
        "update_interval": 1,
        "update_interval_target": 400,
        "target_entropy_ratio": 0.8,
    },
}


def run(args):
    env = gym.make(args.env_id)
    env_test = gym.make(args.env_id)

    algo = DISCRETE_ALGORITHM[args.algo](
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
    p.add_argument("--algo", type=str, default="dqn")
    p.add_argument("--num_steps", type=int, default=50000)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--env_id", type=str, default="CartPole-v0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
