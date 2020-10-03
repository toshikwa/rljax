import os
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter


class Trainer:
    """
    Trainer.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        num_steps=10 ** 6,
        eval_interval=10 ** 4,
        num_eval_episodes=10,
    ):
        super().__init__()

        # Envs.
        self.env = env
        self.env_test = env_test

        # Set seeds.
        self.env.seed(seed)
        self.env_test.seed(2 ** 31 - seed)

        # Algorithm.
        self.algo = algo

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(log_dir, "log.csv")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "summary"))

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()

        for step in range(1, self.num_steps + 1):
            state = self.algo.step(self.env, state)

            if self.algo.is_update():
                self.algo.update(self.writer)

            if step % self.eval_interval == 0:
                self.evaluate(step)

        # Wait for the logging to be finished.
        sleep(2)

    def evaluate(self, step):
        total_return = 0.0
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            while not done:
                action = self.algo.select_action(state)
                state, reward, done, info = self.env_test.step(action)
                total_return += reward
        mean_return = total_return / self.num_eval_episodes

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step)
        print(f"Num steps: {step:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

        # Log to CSV.
        self.log["step"].append(step)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
