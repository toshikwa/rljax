import argparse
import os
from datetime import datetime

from rljax.algorithm import CONTINUOUS_ALGORITHM
from rljax.env.mujoco.metaworld import make_metaworld_env
from rljax.trainer import MetaWorldTrainer

config = {
    "sac": {
        "start_steps": 1000,
        "units_critic": [256, 256, 256],
        "units_actor": [256, 256, 256],
    },
    "sac_discor": {
        "start_steps": 1000,
        "units_critic": [256, 256, 256],
        "units_actor": [256, 256, 256],
        "units_error": [256, 256, 256, 256],
    },
}


def run(args):
    env = make_metaworld_env(args.env_id, seed=args.seed)
    env_test = make_metaworld_env(args.env_id, seed=args.seed)

    algo = CONTINUOUS_ALGORITHM[args.algo](
        num_agent_steps=args.num_agent_steps,
        state_space=env.observation_space,
        action_space=env.action_space,
        seed=args.seed,
        **config[args.algo]
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join("logs", args.env_id, f"{str(algo)}-seed{args.seed}-{time}")

    trainer = MetaWorldTrainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_agent_steps=args.num_agent_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
    )
    trainer.train()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="sac")
    p.add_argument("--num_agent_steps", type=int, default=2 * 10 ** 6)
    p.add_argument("--eval_interval", type=int, default=10000)
    p.add_argument("--env_id", type=str, default="hammer-v1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args)
