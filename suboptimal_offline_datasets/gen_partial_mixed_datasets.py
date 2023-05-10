
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

    return trajs # [(observation, action, reward, next_observation, terminal, timeout)]


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

def sample_segment(traj, min_len, max_len):
  traj_len = len(traj)
  # high can't be negative
  start = np.random.randint(low=0, high=max(traj_len - min_len, 1))
  seg_len = np.random.randint(low=min_len, high=max_len)
  end = min(start + seg_len, traj_len) 
  seg = traj[start:end]
  if (not seg[-1][-2]) and (not seg[-1][-1]): # if not terminal and not timeout, we add articial timeout
    seg[-1][-1] = True
  return seg

def make_env_and_dataset(env_name: str,
                         good_ratio: float,
                         min_len: int = 10,
                         max_len: int = 50,                     
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
      seg = sample_segment(random.choice(good_trajs), min_len, max_len)
      n_good_transitions += len(seg)
      mixed_trajs.append(seg)

    n_bad_transitions = 0
    while n_bad_transitions <= n_transitions - n_good_transitions:
      seg = sample_segment(random.choice(bad_trajs), min_len, max_len)
      n_bad_transitions += len(seg)
      mixed_trajs.append(seg)

    dataset = merge_trajectories(mixed_trajs)

    # Summarize dataset
    # print("Raw dataset")
    # print(f'Keys: {dataset.keys()}, Size: {dataset["observations"].shape[0]}')
    # print("Q-learning dataset")
    # qlearning_dataset = d4rl.qlearning_dataset(env, dataset=dataset)
    # print(f'Keys: {qlearning_dataset.keys()}, Size: {qlearning_dataset["observations"].shape[0]}')
    # print("Sequence dataset")
    # seq_dataset = list(d4rl.sequence_dataset(env, dataset=dataset))
    # print(f"n_trajs: {len(seq_dataset)}")
    # import ipdb; ipdb.set_trace()

    return dataset


if __name__ == "__main__":
    import itertools
    import h5py
    import os
    from tqdm import tqdm

    base_path = "./partial_mixed_datasets"

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

    min_len = 5
    max_len = 500
    for env, level, good_ratio in tqdm(itertools.product(env_names, levels, dataset_ratios)):
        dataset = make_env_and_dataset(f"{env}-{level}-v2", float(good_ratio), min_len=min_len, max_len=max_len)
        file_path = os.path.join(base_path, f"{env}-random-{level}-{good_ratio}-{min_len}-{max_len}-v2.hdf5")
        hf = h5py.File(file_path, 'w')
        for k in dataset.keys():
          hf.create_dataset(k, data=dataset[k], compression='gzip')
        print(f"Save at {file_path}")
        hf.close()
