
import random
import gym
import numpy as np

import d4rl
from tqdm import tqdm


def split_into_trajectories(dataset):
    trajs = [[]]

    for i in tqdm(range(len(dataset["observations"]))):
        trajs[-1].append(([dataset[k][i] for k in ["observations", "actions", "rewards", "next_observations", "terminals", "timeouts"]]))
        if (dataset["terminals"][i] or dataset['timeouts'][i]) and i + 1 < len(dataset["observations"]):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    timeouts = []

    for traj in trajs:
        for (obs, act, rew, next_observation, term, timeout) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_observation)
            terminals.append(term)
            timeouts.append(timeout)

    return {
        "observations": np.stack(observations), 
        "actions": np.stack(actions), 
        "rewards": np.stack(rewards),
        "next_observations": np.stack(next_observations), 
        "terminals": np.stack(terminals), 
        "timeouts": np.stack(timeouts)
        }

def make_env_and_dataset(env_name: str,
                         good_ratio: float,                        
                         ):
    assert good_ratio != 1 and good_ratio != 0

    print("Env:", env_name, "Good ratio:", good_ratio)
    env = gym.make(env_name)
    good_dataset = env.get_dataset()
    good_trajs = split_into_trajectories(good_dataset)

    pure_env_name = env_name.split("-", 1)[0]
    bad_dataset = gym.make(f"{pure_env_name}-random-v2").get_dataset()
    bad_trajs = split_into_trajectories(bad_dataset)

    n_transitions = len(good_dataset["observations"]) # good_dataset.size
    mixed_trajs = []
    n_good_transitions = 0
    while n_good_transitions <= int(good_ratio * n_transitions):
      traj = random.choice(good_trajs)
      n_good_transitions += len(traj)
      mixed_trajs.append(traj)

    n_bad_transitions = 0
    while n_bad_transitions <= n_transitions - n_good_transitions:
      traj = random.choice(bad_trajs)
      n_bad_transitions += len(traj)
      mixed_trajs.append(traj)

    dataset = merge_trajectories(mixed_trajs)


    return dataset


if __name__ == "__main__":
    import itertools
    import h5py
    import os
    from tqdm import tqdm

    base_path = "./custom_datasets"

    env_names = [
      "ant",
      "hopper",
      "walker2d",
      "halfcheetah",
    ]

    levels = [
       "medium",
       "expert",
    ]

    dataset_ratios = [
        "0.01",
        "0.05",
        "0.1",
        "0.25",
        "0.5",
    ]

    for env, level, good_ratio in tqdm(itertools.product(env_names, levels, dataset_ratios)):
        dataset = make_env_and_dataset(f"{env}-{level}-v2", float(good_ratio))
        file_path = os.path.join(base_path, f"{env}-random-{level}-{good_ratio}-v2.hdf5")
        hf = h5py.File(file_path, 'w')
        for k in dataset.keys():
          hf.create_dataset(k, data=dataset[k], compression='gzip')
        print(f"Save at {file_path}")
        hf.close()
