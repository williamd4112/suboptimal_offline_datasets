Suboptimal offline RL datasets
=

This repo curates `random-medium-x%-v2` and `random-expert-x%-v2` datasets based on Mujoco datasets in D4RL. The datasets are used in [Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting ](https://openreview.net/pdf?id=OhUAblg27z), ICLR 2023.

# Installation
- Download datasets from https://drive.google.com/file/d/1KRfeHpqcSI2gslhxx6IrurqSGpSBSXE9/view?usp=share_link
- Unzip the dataset and extract to `suboptimal_offline_datasets/custom_datasets` (you should see lots of `*.hdf5` files in this directory)
- `pip install -e .`


# Usage
Before you call `gym.make`, be sure to do `import suboptimal_offline_datasets`. The following is an example:
```
import gym
import suboptimal_offline_datasets

env = gym.make("ant-random-medium-0.5-v2")
dataset = env.get_dataset()
```

# Datasets (environment) list
```
{ant,halfcheetah,hopper,walker2d}-random-{medium,expert}-{0.01,0.05,0.1,0.25,0.5}-v2
```
