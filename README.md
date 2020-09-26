**WARNING: Rljax is currently in a beta version and being actively improved. Any contributions are welcome :)**

# RL Algorithms in JAX
Rljax is a collection of RL algorithms written in JAX.

## Setup
You can install dependencies simply by executing the following. To use GPUs, nvidia-driver and CUDA must be installed.
```bash
pip install --upgrade https://storage.googleapis.com/jax-releases/`nvcc -V | sed -En "s/.* release ([0-9]*)\.([0-9]*),.*/cuda\1\2/p"`/jaxlib-0.1.55-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl jax
pip install -e .
```

If you don't have a GPU, please executing the following instead.
```bash
pip install --upgrade jaxlib jax
pip install -e .
```

If you want to use a [MuJoCo](http://mujoco.org/) physics engine, please install [mujoco-py](https://github.com/openai/mujoco-py).

## Algorithms
Currently, following algorithms have been implemented.

- [x] Proximal Policy Optimization(PPO)
- [x] Deep Deterministic Policy Gradient(DDPG)
- [x] Twin Delayed DDPG(TD3)
- [x] Soft Actor-Critic(SAC)
- [x] Deep Q Network(DQN)
- [x] N-step return
- [x] Dueling Network
- [x] Double Q-Learning
- [x] Prioritized Experience Replay(PER)
- [x] Soft Actor-Critic for Discrete Settings(SAC-Discrete)

We plan to implement the following algorithms in the future.

- [ ] Quantile Regression DQN(QR-DQN)
- [ ] Implicit Quantile Network(IQN)

Below shows that our algorithms successfully learning the discrete action environment `CartPole-v0` and the continuous action environment `InvertedPendulum-v2`.

<img src="https://user-images.githubusercontent.com/37267851/94338069-7bc3fb80-002a-11eb-91ae-d163ebebd2a7.png" title="CartPole-v0" width=400><img src="https://user-images.githubusercontent.com/37267851/94338071-7e265580-002a-11eb-8f10-b8cf5dac0207.png" title="InvertedPendulum-v2" width=400>