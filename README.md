**WARNING: Rljax is currently in a beta version and being actively improved. Any contributions are welcome :)**

# RL Algorithms in JAX
Rljax is a collection of RL algorithms written in JAX.

## Setup
You can install dependencies simply by executing the following. To use GPUs, CUDA (10.0, 10.1, 10.2 or 11.0) must be installed.
```bash
pip install https://storage.googleapis.com/jax-releases/`nvcc -V | sed -En "s/.* release ([0-9]*)\.([0-9]*),.*/cuda\1\2/p"`/jaxlib-0.1.55-`python3 -V | sed -En "s/Python ([0-9]*)\.([0-9]*).*/cp\1\2/p"`-none-manylinux2010_x86_64.whl jax==0.2.0
pip install -e .
```

If you don't have a GPU, please executing the following instead.
```bash
pip install jaxlib==0.1.55 jax==0.2.0
pip install -e .
```

If you want to use a [MuJoCo](http://mujoco.org/) physics engine, please install [mujoco-py](https://github.com/openai/mujoco-py).
```bash
pip install mujoco_py==2.0.2.11
```

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
All algorithms can be trained in a few lines of codes.

<details>
<summary>Getting started</summary>

Here is a quick example of how to train DQN on `CartPole-v0`.

```Python
import gym

from rljax.algorithm import DQN
from rljax.trainer import Trainer

NUM_STEPS = 20000
SEED = 0

env = gym.make("CartPole-v0")
env_test = gym.make("CartPole-v0")

algo = DQN(
    num_steps=NUM_STEPS,
    state_space=env.observation_space,
    action_space=env.action_space,
    seed=SEED,
    batch_size=256,
    start_steps=1000,
    update_interval=1,
    update_interval_target=400,
)

trainer = Trainer(
    env=env,
    env_test=env_test,
    algo=algo,
    log_dir="/tmp/rljax/dqn",
    num_steps=NUM_STEPS,
    eval_interval=1000,
    seed=SEED,
)
trainer.train()
```

</details>

<details>
<summary>Simple examples</summary>

Below shows that our algorithms successfully learning the discrete action environment `CartPole-v0` ([code](https://github.com/ku2482/rljax/blob/master/examples/train_continuous_easy.py)) and the continuous action environment `InvertedPendulum-v2` ([code](https://github.com/ku2482/rljax/blob/master/examples/train_discrete_easy.py)).

<img src="https://user-images.githubusercontent.com/37267851/94864541-1da67680-0477-11eb-97ce-c6abc0eb2c51.png" title="CartPole-v0" width=400><img src="https://user-images.githubusercontent.com/37267851/94751929-c5af3780-03c4-11eb-8372-832762d8dfc1.png" title="InvertedPendulum-v2" width=400>

</details>

<details>
<summary>MuJoCo(Gym)</summary>

I benchmarked my implementations in environments from MuJoCo's `-v3` task suites, following [Spinning Up's benchmarks](https://spinningup.openai.com/en/latest/spinningup/bench.html) ([code](https://github.com/ku2482/rljax/blob/master/examples/train_mujoco.py)).

<img src="https://user-images.githubusercontent.com/37267851/94887999-b0b0d200-04b2-11eb-9a37-7e2b87dfa71a.png" title="HalfCheetah-v3" width=400><img src="https://user-images.githubusercontent.com/37267851/94888002-b1e1ff00-04b2-11eb-87da-243f39d325b6.png" title="Walker2d-v3" width=400>

</details>
