# RL Algorithms in JAX
Rljax is a JAX implementation of RL algorithms.

Currently, following algorithms have been implemented.

- [x] Deep Deterministic Policy Gradient(DDPG)
- [x] Twin Delayed DDPG(TD3)
- [x] Soft Actor-Critic(SAC)

We are planning to implement more algorithms.

- [ ] Deep Q Network(DQN) and extensions
- [ ] Quantile Regression DQN(QR-DQN)
- [ ] Implicit Quantile Network(IQN)
- [ ] Soft Actor-Critic for Discrete Settings(SAC-Discrete)


## Setup

```bash
# Install jaxlib and jax for your environment.
pip install --upgrade https://storage.googleapis.com/jax-releases/`nvcc -V | sed -En "s/.* release ([0-9]*)\.([0-9]*),.*/cuda\1\2/p"`/jaxlib-0.1.55-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl jax
# Install other dependencies.
pip install -r requirements.txt
```

If you want to use a [MuJoCo](http://mujoco.org/) physics engine, please install [mujoco-py](https://github.com/openai/mujoco-py).