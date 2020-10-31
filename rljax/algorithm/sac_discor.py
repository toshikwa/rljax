import jax.numpy as jnp

from rljax.algorithm.misc import DisCorMixIn
from rljax.algorithm.sac import SAC
from rljax.util import optimize


class SAC_DisCor(DisCorMixIn, SAC):
    name = "SAC+DisCor"

    def __init__(
        self,
        num_agent_steps,
        state_space,
        action_space,
        seed,
        max_grad_norm=None,
        gamma=0.99,
        num_critics=2,
        buffer_size=10 ** 6,
        batch_size=256,
        start_steps=10000,
        update_interval=1,
        tau=5e-3,
        fn_actor=None,
        fn_critic=None,
        fn_error=None,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        lr_error=3e-4,
        units_actor=(256, 256),
        units_critic=(256, 256),
        units_error=(256, 256, 256),
        log_std_min=-20.0,
        log_std_max=2.0,
        d2rl=False,
        init_alpha=1.0,
        init_error=10.0,
        adam_b1_alpha=0.9,
    ):
        SAC.__init__(
            self,
            num_agent_steps=num_agent_steps,
            state_space=state_space,
            action_space=action_space,
            seed=seed,
            max_grad_norm=max_grad_norm,
            gamma=gamma,
            nstep=1,
            num_critics=num_critics,
            buffer_size=buffer_size,
            use_per=False,
            batch_size=batch_size,
            start_steps=start_steps,
            update_interval=update_interval,
            tau=tau,
            fn_actor=fn_actor,
            fn_critic=fn_critic,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            lr_alpha=lr_alpha,
            units_actor=units_actor,
            units_critic=units_critic,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            d2rl=d2rl,
            init_alpha=init_alpha,
            adam_b1_alpha=adam_b1_alpha,
        )
        DisCorMixIn.__init__(
            self,
            num_critics=num_critics,
            fn_error=fn_error,
            units_error=units_error,
            d2rl=d2rl,
            init_error=init_error,
        )

    def update(self, writer=None):
        self.learning_step += 1
        _, batch = self.buffer.sample(self.batch_size)
        state, action, reward, done, next_state = batch

        # Calculate weights.
        weight_list = self._calculate_weight_list(
            params_actor=self.params_actor,
            params_error_target=self.params_error_target,
            rm_error_list=self.rm_error_list,
            done=done,
            next_state=next_state,
            key=next(self.rng),
        )

        # Update critic.
        self.opt_state_critic, self.params_critic, loss_critic, abs_td_list = optimize(
            self._loss_critic,
            self.opt_critic,
            self.opt_state_critic,
            self.params_critic,
            self.max_grad_norm,
            params_critic_target=self.params_critic_target,
            params_actor=self.params_actor,
            log_alpha=self.log_alpha,
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
            weight=weight_list,
            **self.kwargs_critic,
        )

        # Update error model.
        self.opt_state_error, self.params_error, loss_error, mean_error_list = optimize(
            self._loss_error,
            self.opt_error,
            self.opt_state_error,
            self.params_error,
            self.max_grad_norm,
            params_error_target=self.params_error_target,
            params_actor=self.params_actor,
            state=state,
            action=action,
            done=done,
            next_state=next_state,
            abs_td_list=abs_td_list,
            key=next(self.rng),
        )

        # Update actor.
        self.opt_state_actor, self.params_actor, loss_actor, mean_log_pi = optimize(
            self._loss_actor,
            self.opt_actor,
            self.opt_state_actor,
            self.params_actor,
            self.max_grad_norm,
            params_critic=self.params_critic,
            log_alpha=self.log_alpha,
            state=state,
            **self.kwargs_actor,
        )

        # Update alpha.
        self.opt_state_alpha, self.log_alpha, loss_alpha, _ = optimize(
            self._loss_alpha,
            self.opt_alpha,
            self.opt_state_alpha,
            self.log_alpha,
            None,
            mean_log_pi=mean_log_pi,
        )

        # Update target networks.
        self.params_critic_target = self._update_target(self.params_critic_target, self.params_critic)
        self.params_error_target = self._update_target(self.params_error_target, self.params_error)
        self.rm_error_list = self._update_target(self.rm_error_list, mean_error_list)

        if writer and self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic, self.learning_step)
            writer.add_scalar("loss/actor", loss_actor, self.learning_step)
            writer.add_scalar("loss/alpha", loss_alpha, self.learning_step)
            writer.add_scalar("loss/error", loss_error, self.learning_step)
            writer.add_scalar("stat/alpha", jnp.exp(self.log_alpha), self.learning_step)
            writer.add_scalar("stat/entropy", -mean_log_pi, self.learning_step)
