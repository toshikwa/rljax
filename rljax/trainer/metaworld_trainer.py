import numpy as np
import pandas as pd

from .base_trainer import Trainer


class MetaWorldTrainer(Trainer):
    """
    Trainer for MetaWorld's task suite.
    """

    def evaluate(self, step):
        total_return = 0.0
        success = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            done = False
            while not done:
                action = self.algo.select_action(state)
                state, reward, done, info = self.env_test.step(action)
                total_return += reward
                if "success" in info.keys():
                    success.append(float(info["success"]))

        # Log success rate.
        if len(success) > 0:
            success_rate = np.mean(success)
            # To TensorBoard.
            self.writer.add_scalar("success_rate/test", success_rate, step * self.action_repeat)
            # To CSV.
            if "success_rate" not in self.log.keys():
                self.log["success_rate"] = [success_rate]
            else:
                self.log["success_rate"].append(success_rate)

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        # To TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step * self.action_repeat)
        # To CSV.
        self.log["step"].append(step * self.action_repeat)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to standard output.
        print(f"Num steps: {step * self.action_repeat:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")
