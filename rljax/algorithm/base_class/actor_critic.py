import os
from abc import abstractmethod
from functools import partial
from typing import List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from rljax.algorithm.base_class.base_algoirithm import OffPolicyAlgorithm, OnPolicyAlgorithm
from rljax.util import fake_action, fake_state, load_params, save_params


class ActorCriticMixIn:
    """ MixIn for Actor-Critic algorithms. """

    def __init__(self):
        # If _loss_critic() method uses random key or not.
        if not hasattr(self, "use_key_critic"):
            self.use_key_critic = False
        # If _loss_actor() method uses random key or not.
        if not hasattr(self, "use_key_actor"):
            self.use_key_actor = False

    @property
    def kwargs_critic(self):
        return {"key": next(self.rng)} if self.use_key_critic else {}

    @property
    def kwargs_actor(self):
        return {"key": next(self.rng)} if self.use_key_actor else {}

    def select_action(self, state):
        action = self._select_action(self.params_actor, state[None, ...])
        return np.array(action[0])

    @abstractmethod
    def _select_action(self, params_actor, state):
        pass

    @abstractmethod
    def _explore(self, params_actor, state, key):
        pass

    def save_params(self, save_dir):
        save_params(self.params_critic, os.path.join(save_dir, "params_critic.npz"))
        save_params(self.params_actor, os.path.join(save_dir, "params_actor.npz"))

    def load_params(self, save_dir):
        self.params_critic = load_params(os.path.join(save_dir, "params_critic.npz"))
        self.params_actor = load_params(os.path.join(save_dir, "params_actor.npz"))


class OnPolicyActorCritic(ActorCriticMixIn, OnPolicyAlgorithm):
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
        ActorCriticMixIn.__init__(self)
        OnPolicyAlgorithm.__init__(
            self,
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            self.fake_args_critic = (fake_state(state_space),)
        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space),)

    def is_update(self):
        return self.agent_step % self.buffer_size == 0

    def explore(self, state):
        action, log_pi = self._explore(self.params_actor, state[None, ...], next(self.rng))
        return np.array(action[0]), np.array(log_pi[0])


class OffPolicyActorCritic(ActorCriticMixIn, OffPolicyAlgorithm):
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
        num_critics,
        buffer_size,
        use_per,
        batch_size,
        start_steps,
        update_interval,
        update_interval_target=None,
        tau=None,
    ):
        ActorCriticMixIn.__init__(self)
        OffPolicyAlgorithm.__init__(
            self,
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
        self.num_critics = num_critics
        # Define fake input for critic.
        if not hasattr(self, "fake_args_critic"):
            if self.discrete_action:
                self.fake_args_critic = (fake_state(state_space),)
            else:
                self.fake_args_critic = (fake_state(state_space), fake_action(action_space))
        # Define fake input for actor.
        if not hasattr(self, "fake_args_actor"):
            self.fake_args_actor = (fake_state(state_space),)

    def explore(self, state):
        action = self._explore(self.params_actor, state[None, ...], next(self.rng))
        return np.array(action[0])

    @abstractmethod
    def _sample_action(self, params_actor, state, *args, **kwargs):
        pass

    @partial(jax.jit, static_argnums=0)
    def _calculate_value_list(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> List[jnp.ndarray]:
        return self.critic.apply(params_critic, state, action)

    @partial(jax.jit, static_argnums=0)
    def _calculate_value(
        self,
        params_critic: hk.Params,
        state: np.ndarray,
        action: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.asarray(self._calculate_value_list(params_critic, state, action)).min(axis=0)

    @abstractmethod
    def _calculate_target(self, params_critic_target, reward, done, next_state, next_action, *args, **kwargs):
        pass

    @partial(jax.jit, static_argnums=0)
    def _calculate_loss_critic_and_abs_td(
        self,
        value_list: List[jnp.ndarray],
        target: jnp.ndarray,
        weight: np.ndarray,
    ) -> jnp.ndarray:
        abs_td = jnp.abs(target - value_list[0])
        loss_critic = (jnp.square(abs_td) * weight).mean()
        for value in value_list[1:]:
            loss_critic += (jnp.square(target - value) * weight).mean()
        return loss_critic, jax.lax.stop_gradient(abs_td)
