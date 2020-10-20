import argparse
import os
from datetime import datetime

from rljax.algorithm import SLAC
from rljax.env.mujoco.dmc import make_dmc_env
from rljax.trainer import SLACTrainer


def run(args):
    env = make_dmc_env(args.domain_name, args.task_name, args.action_repeat, 1, 64)
    env_test = make_dmc_env(args.domain_name, args.task_name, args.action_repeat, 1, 64)

    algo = SLAC(
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        num_sequences=args.num_sequences,
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", f"{args.domain_name}-{args.task_name}", f"SLAC-seed{args.seed}-{time}")

    trainer = SLACTrainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        action_repeat=args.action_repeat,
        num_sequences=args.num_sequences,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_agent_steps", type=int, default=500000)
    p.add_argument("--eval_interval", type=int, default=5000)
    p.add_argument('--domain_name', type=str, default='cheetah')
    p.add_argument('--task_name', type=str, default='run')
    p.add_argument('--action_repeat', type=int, default=4)
    p.add_argument('--num_sequences', type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
