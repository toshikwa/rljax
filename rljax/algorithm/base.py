from abc import ABC, abstractmethod
from functools import partial

import jax
from haiku import PRNGSequence
from rljax.buffer import PrioritizedReplayBuffer, ReplayBuffer, RolloutBuffer
from rljax.utils import soft_update


class Algorithm(ABC):
    """
    Base class for algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
    ):
        self.rng = PRNGSequence(seed)
        self.learning_steps = 0
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma

    @abstractmethod
    def is_update(self, step):
        pass

    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def step(self, env, state, t, step):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class OffPolicyAlgorithm(Algorithm):
    """
    Base class for off-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        nstep,
        buffer_size,
        use_per,
    ):
        super(OffPolicyAlgorithm, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
        )

        self.discount = gamma ** nstep
        self.use_per = use_per
        self.buffer = (PrioritizedReplayBuffer if use_per else ReplayBuffer)(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
            gamma=gamma,
            nstep=nstep,
        )


class OnPolicyAlgorithm(Algorithm):
    """
    Base class for on-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        buffer_size,
    ):
        super(OnPolicyAlgorithm, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
        )

        self.discount = gamma
        self.buffer_size = buffer_size
        self.buffer = RolloutBuffer(
            buffer_size=buffer_size,
            state_space=state_space,
            action_space=action_space,
        )


class ContinuousOffPolicyAlgorithm(OffPolicyAlgorithm):
    """
    Base class for continuous off-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        tau,
    ):
        super(ContinuousOffPolicyAlgorithm, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
        )

        self.batch_size = batch_size
        self.start_steps = start_steps
        self._update_target = jax.jit(partial(soft_update, tau=tau))

    def is_update(self, step):
        return step >= self.start_steps

    def step(self, env, state, t, step):
        t += 1

        if step <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, next_state, done)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t


class DiscreteOffPolicyAlgorithm(OffPolicyAlgorithm):
    """
    Base class for discrete off-policy algorithms.
    """

    def __init__(
        self,
        state_space,
        action_space,
        seed,
        gamma,
        nstep,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target,
    ):
        super(DiscreteOffPolicyAlgorithm, self).__init__(
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            gamma=gamma,
            nstep=nstep,
            buffer_size=buffer_size,
            use_per=use_per,
        )

        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_interval = update_interval
        self.update_interval_target = update_interval_target
        self._update_target = jax.jit(partial(soft_update, tau=1.0))

    def is_update(self, step):
        return step % self.update_interval == 0 and step >= self.start_steps


class ContinuousOnPolicyAlgorithm(OnPolicyAlgorithm):
    """
    Base class for continuous on-policy algorithms.
    """

    def is_update(self, step):
        return step % self.buffer_size == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t
