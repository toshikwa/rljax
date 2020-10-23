import os
from collections import deque
from time import sleep, time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .base_trainer import Trainer


class SlacInput:
    """
    Input for SLAC.
    """

    def __init__(self, state_space, action_space, num_sequences):
        self.state_shape = state_space.shape
        self.action_shape = action_space.shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state, dtype=np.uint8)[None, ...]

    @property
    def action(self):
        return np.array(self._action, dtype=np.float32).reshape(1, -1)


class SLACTrainer(Trainer):
    """
    Trainer for SLAC.
    """

    def __init__(
        self,
        env,
        env_test,
        algo,
        log_dir,
        seed=0,
        action_repeat=1,
        num_sequences=8,
        num_agent_steps=10 ** 6,
        eval_interval=10 ** 4,
        num_eval_episodes=10,
        save_params=False,
    ):
        super(SLACTrainer, self).__init__(
            env=env,
            env_test=env_test,
            algo=algo,
            log_dir=log_dir,
            seed=seed,
            action_repeat=action_repeat,
            num_agent_steps=num_agent_steps,
            eval_interval=eval_interval,
            num_eval_episodes=num_eval_episodes,
            save_params=save_params,
        )

        # Inputs for training and evaluation.
        self.input = SlacInput(env.observation_space, env.action_space, num_sequences)
        self.input_test = SlacInput(env.observation_space, env.action_space, num_sequences)

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Initialize the environment.
        state = self.env.reset()
        self.input.reset_episode(state)
        self.algo.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        for step in range(1, self.algo.start_steps + 1):
            self.algo.step(self.env, self.input)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.algo.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.algo.update_latent(self.writer)

        for step in range(1, self.num_agent_steps + 1):
            self.algo.step(self.env, self.input)

            if self.algo.is_update():
                self.algo.update_latent(self.writer)
                self.algo.update_sac(self.writer)

            if step % self.eval_interval == 0:
                self.evaluate(step)
                if self.save_params:
                    self.algo.save_params(os.path.join(self.param_dir, f"step{step}"))

        # Wait for the logging to be finished.
        sleep(2)

    def evaluate(self, step):
        total_return = 0.0

        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            self.input_test.reset_episode(state)
            done = False
            while not done:
                action = self.algo.select_action(self.input_test)
                state, reward, done, _ = self.env_test.step(action)
                self.input_test.append(state, action)
                total_return += reward

        # Log mean return.
        mean_return = total_return / self.num_eval_episodes
        # To TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step * self.action_repeat)
        # To CSV.
        self.log["step"].append(step * self.action_repeat)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to standard output.
        print(f"Num steps: {step * self.action_repeat:<6}   Return: {mean_return:<5.1f}   Time: {self.time}")
