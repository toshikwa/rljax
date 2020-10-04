import argparse
import os
from datetime import datetime

from rljax.algorithm import SAC_AE
from rljax.env.contrib import make_dmc_env
from rljax.trainer import Trainer


def run(args):
    env = make_dmc_env(args.domain_name, args.task_name, args.action_repeat)
    env_test = make_dmc_env(args.domain_name, args.task_name, args.action_repeat)

    algo = SAC_AE(
        num_steps=args.num_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", f"{args.domain_name}-{args.task_name}", f"SAC+AE-seed{args.seed}-{time}")

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        action_repeat=args.action_repeat,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_steps", type=int, default=750000)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument('--domain_name', type=str, default='cheetah')
    p.add_argument('--task_name', type=str, default='run')
    p.add_argument('--action_repeat', type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
