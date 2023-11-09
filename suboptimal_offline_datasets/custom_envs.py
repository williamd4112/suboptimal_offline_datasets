import gym
import numpy as np
from d4rl import offline_env
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
from d4rl.utils.wrappers import NormalizedBoxEnv

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
