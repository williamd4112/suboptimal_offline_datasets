import gym
import numpy as np
from d4rl import offline_env
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
from d4rl.utils.wrappers import NormalizedBoxEnv

import neorl

class D4RLNeoRLEnv(gym.Env):
    
    data_type: str

    def __init__(self, env, data_type, **kwargs):
        super().__init__()
        self.env = env
        self.data_type = data_type
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        try:
            self.env.close()
        except:
            print("Env doesn't have close method.")

    def get_normalized_score(self, score):
        # TODO: provide normalized score
        return score

    def get_dataset(self, **kwargs):
        train_data, _ = self.env.get_dataset(
            data_type=self.data_type, 
            need_val=False, **kwargs)
        train_data["observations"] = train_data.pop("obs")
        train_data["next_observations"] = train_data.pop("next_obs")
        train_data["actions"] = train_data.pop("action")
        train_data["rewards"] = train_data.pop("reward")
        train_data["terminals"] = train_data.pop("done")
        train_data["timeouts"] = np.zeros_like(train_data["terminals"])

        return train_data

def make_neorl_env(task, data_type):
    env = neorl.make(task)
    env = D4RLNeoRLEnv(env, data_type=data_type)
    return env

class OfflineAntEnv(AntEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        AntEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

    def get_dataset(self, h5path=None):
        return super().get_dataset(self.dataset_url)

class OfflineHopperEnv(HopperEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HopperEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)
    
    def get_dataset(self, h5path=None):
        return super().get_dataset(self.dataset_url)

class OfflineHalfCheetahEnv(HalfCheetahEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)
    
    def get_dataset(self, h5path=None):
        return super().get_dataset(self.dataset_url)

class OfflineWalker2dEnv(Walker2dEnv, offline_env.OfflineEnv):
    def __init__(self, **kwargs):
        Walker2dEnv.__init__(self,)
        offline_env.OfflineEnv.__init__(self, **kwargs)

    def get_dataset(self, h5path=None):
        return super().get_dataset(self.dataset_url)


def get_ant_env(**kwargs):
    return NormalizedBoxEnv(OfflineAntEnv(**kwargs))

def get_cheetah_env(**kwargs):
    return NormalizedBoxEnv(OfflineHalfCheetahEnv(**kwargs))

def get_hopper_env(**kwargs):
    return NormalizedBoxEnv(OfflineHopperEnv(**kwargs))

def get_walker_env(**kwargs):
    return NormalizedBoxEnv(OfflineWalker2dEnv(**kwargs))
