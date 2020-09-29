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
- [x] Quantile Regression DQN(QR-DQN)
- [x] Implicit Quantile Network(IQN)
- [x] Soft Actor-Critic for Discrete Settings(SAC-Discrete)

## Examples

Below shows that our algorithms successfully learning the discrete action environment `CartPole-v0` and the continuous action environment `InvertedPendulum-v2`. Note that while other discrete algorithms use two linear layers, IQN uses three linear layers due to the structure of the algorithm.

<img src="https://user-images.githubusercontent.com/37267851/94482857-e59ff900-0214-11eb-89e7-5c53f5fecc14.png" title="CartPole-v0" width=400><img src="https://user-images.githubusercontent.com/37267851/94509799-1d786200-0250-11eb-8a0e-779cd76e9dc3.png" title="InvertedPendulum-v2" width=400>
