import os
from abc import ABC, abstractmethod
from functools import partial

import jax
import numpy as np
from gym.spaces import Box
from haiku import PRNGSequence

from rljax.buffer import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer, SLACReplayBuffer
from rljax.util import load_params, save_params, soft_update


class Algorithm(ABC):
    """
    Base class for algorithms.
    """

    name = None

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
    ):
        np.random.seed(seed)
        self.rng = PRNGSequence(seed)

        self.agent_step = 0
        self.episode_step = 0
        self.learning_step = 0
        self.num_agent_steps = num_agent_steps
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        # Define fake input for JIT.
        self.fake_state = state_space.sample()[None, ...]
        if len(state_space.shape) == 1:
            self.fake_state = self.fake_state.astype(np.float32)
        if type(action_space) == Box:
            self.discrete_action = False
            self.fake_action = action_space.sample()[None, ...]
            self.fake_action = self.fake_action.astype(np.float32)
        else:
            self.discrete_action = True
            self.fake_action = None

    def get_mask(self, env, done):
        return done if self.episode_step != env._max_episode_steps or self.discrete_action else False

    @abstractmethod
    def is_update(self):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def step(self, env, state):
        pass

    @abstractmethod
    def save_params(self, save_dir):
        pass

    @abstractmethod
    def load_params(self, save_dir):
        pass

    def __str__(self):
        return self.name


class OnPolicyActorCritic(Algorithm):
    """
    Base class for on-policy Actor-Critic algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        buffer_size,
        batch_size,
    ):
        super(OnPolicyActorCritic, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
        )
        self.discount = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action, log_pi = self._explore(self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0]), np.array(log_pi[0])

    @abstractmethod
    def _select_action(self, params_actor, state):
        pass

    @abstractmethod
    def _explore(self, params_actor, key, state):
        pass

    def is_update(self):
        return self.agent_step % self.buffer_size == 0

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = self.get_mask(env, done)
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            self.episode_step = 0
            next_state = env.reset()

        return next_state

    @abstractmethod
    def update(self, writer):
        pass

    def save_params(self, save_dir):
        save_params(self.params_critic, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_critic = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = load_params(os.path.join(save_dir, "params_actor.npz"))


class OffPolicyAlgorithm(Algorithm):
    """
    Base class for off-policy algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        assert update_interval_target or tau
        super(OffPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )
        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                buffer_size=buffer_size,
                state_space=state_space,
                action_space=action_space,
                gamma=gamma,
                nstep=nstep,
                beta_steps=(num_agent_steps - start_steps) / update_interval,
            )
        else:
            self.buffer = ReplayBuffer(
                buffer_size=buffer_size,
                state_space=state_space,
                action_space=action_space,
                gamma=gamma,
                nstep=nstep,
            )

        self.discount = gamma ** nstep
        self.use_per = use_per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target

        if update_interval_target:
            self._update_target = jax.jit(partial(soft_update, tau=1.0))
        else:
            self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self):
        return self.agent_step % self.update_interval == 0 and self.agent_step >= self.start_steps

    @abstractmethod
    def explore(self, state):
        pass

    def step(self, env, state):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, done, _ = env.step(action)
        mask = self.get_mask(env, done)
        self.buffer.append(state, action, reward, mask, next_state, done)

        if done:
            self.episode_step = 0
            next_state = env.reset()

        return next_state

    @abstractmethod
    def update(self, writer):
        pass


class OffPolicyActorCritic(OffPolicyAlgorithm):
    """
    Base class for off-policy Actor-Critic algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        super(OffPolicyActorCritic, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
            tau=tau,
        )
        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            if self.discrete_action:
                self.fake_args_critic = (self.fake_state,)
            else:
                self.fake_args_critic = (self.fake_state, self.fake_action)

        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (self.fake_state,)

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    def explore(self, state):
        action = self._explore(self.params_actor, next(self.rng), state[None, ...])
        return np.array(action[0])

    @abstractmethod
    def _select_action(self, params_actor, state):
        pass

    @abstractmethod
    def _explore(self, params_actor, key, state):
        pass

    def save_params(self, save_dir):
        save_params(self.params_critic, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_critic = self.params_critic_target = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = load_params(os.path.join(save_dir, "params_actor.npz"))


class QLearning(OffPolicyAlgorithm):
    """
    Base class for discrete Q-learning algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target,
        eps,
        eps_eval,
        eps_decay_steps,
    ):
        super(QLearning, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            update_interval_target=update_interval_target,
        )
        self.eps = eps
        self.eps_eval = eps_eval
        self.eps_decay_steps = eps_decay_steps
        if not hasattr(self, "fake_args"):
            self.fake_args = (self.fake_state,)

    def select_action(self, state):
        if np.random.rand() < self.eps_eval:
            action = self.action_space.sample()
        else:
            action = self.forward(state[None, ...])
            action = np.array(action[0])
        return action

    def explore(self, state):
        if np.random.rand() < self.eps_train:
            action = self.action_space.sample()
        else:
            action = self.forward(state[None, ...])
            action = np.array(action[0])
        return action

    def forward(self, state):
        return self._forward(self.params, state)

    @abstractmethod
    def _forward(self, params, state):
        pass

    @property
    def eps_train(self):
        if self.agent_step > self.eps_decay_steps:
            return self.eps
        return 1.0 + (self.eps - 1.0) / self.eps_decay_steps * self.agent_step

    def save_params(self, save_dir):
        save_params(self.params, os.path.join(save_dir, "params.npz"))

    def load_params(self, save_dir):
        self.params = self.params_target = load_params(os.path.join(save_dir, "params.npz"))


class SlacAlgorithm(Algorithm):
    """
    Base class for SLAC-based algorithms.
    """

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm,
        gamma,
        num_sequences,
        buffer_size,
        batch_size_sac,
        batch_size_model,
        start_steps,
        initial_learning_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        assert update_interval_target or tau
        super(OffPolicyAlgorithm, self).__init__(
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
        )

        self.buffer = SLACReplayBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            num_sequences=num_sequences,
        )

        self.learning_step_model = 0
        self.learning_step_sac = 0
        self.discount = gamma
        self.num_sequences = num_sequences
        self.batch_size_sac = batch_size_sac
        self.batch_size_model = batch_size_model
        self.start_steps = start_steps
        self.initial_learning_steps = initial_learning_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target

        if update_interval_target:
            self._update_target = jax.jit(partial(soft_update, tau=1.0))
        else:
            self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self):
        return self.agent_step % self.update_interval == 0 and self.agent_step >= self.start_steps

    def select_action(self, ob):
        feature_action = self._preprocess(self.params_model, ob.state, ob.action)
        action = self._select_action(self.params_actor, feature_action)
        return np.array(action[0])

    def explore(self, ob):
        feature_action = self._preprocess(self.params_model, ob.state, ob.action)
        action = self._explore(self.params_actor, next(self.rng), feature_action)
        return np.array(action[0])

    def step(self, env, ob):
        self.agent_step += 1
        self.episode_step += 1

        if self.agent_step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)

        state, reward, done, _ = env.step(action)
        ob.append(state, action)
        mask = self.get_mask(env, done)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            self.episode_step = 0
            state = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return None

    def update(self, writer):
        NotImplementedError

    @abstractmethod
    def update_model(self, writer):
        pass

    @abstractmethod
    def update_sac(self, writer):
        pass
