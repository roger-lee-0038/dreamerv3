from typing import Optional
import numpy as np
import gymnasium as gym
import math
import sys
sys.path.insert(0,"/export/home/liwangzhen/Research/Hierarchical")
from SAR_Behaviour import SAR_Behaviour


class SAR10BitEnv1(gym.Env):

    def __init__(self, init_num=1, max_iter=100, index=0):
        self.sar = SAR_Behaviour()
        self.sar.set_name_suffix(f"v1env{index}")
        
        # The number of objectives
        self.n_objs = 3
        self.n_mets = 9
        # The number of initial points
        self.init_num = init_num
        # The maximal number of iterations
        self.max_iter = max_iter
        # The counter for maximal number of iterations
        self.termination_cnt = 0

        self.current_action = None
        self.current_obs = None
        self.weights = np.random.rand(self.n_objs)
        self.weights = self.weights / self.weights.sum()
        self.current_val = np.inf

        self.observation_space = gym.spaces.Box(
            low = -1e30,
            high = 1e30,
            shape = (self.init_num, self.n_mets + self.n_objs),
            dtype = np.float32,
        )

        self.action_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (self.init_num, 3 + self.sar.in_dim),
            dtype = np.float32,
        )

    def _get_action(self, action, initial=False):
        if initial:
            return action
        else:
            return np.clip(
                        self.current_action + 2*action - 1,
                        0,
                        1,
                    )
    
    def _get_obs(self, action):
        obs = []
        for sub_action in action:
            sarResults = self.sar(sub_action)
            obs.append(list(sarResults.values()) + self.weights.tolist())
        return np.array(obs, dtype=np.float32)

    def _get_reward(self, obs):
        nor_weights = self.weights * np.array([1, 1e4, 1e9])
        val = np.average(np.tensordot(obs[:, :3], nor_weights, axes=([-1], [-1])))
        reward = self.current_val - val
        self.current_val = val
        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if seed is not None:
            #self.observation_space.seed(seed)
            self.action_space.seed(seed)
        #if options.get("max_iter"):
        #    self.max_iter = options.get("max_iter")
        self.termination_cnt = 0

        # Initialize
        self.weights = np.random.rand(self.n_objs)
        self.weights = self.weights / self.weights.sum()

        self.current_action = self.action_space.sample()
        self.current_obs = self._get_obs(self.current_action)

        nor_weights = self.weights * np.array([1, 1e4, 1e9])
        val = np.average(np.tensordot(self.current_obs[:, :3], self.weights, axes=([-1], [-1])))
        self.current_val = val

        info = {"current_obs": self.current_obs, "current_action": self.current_action, "current_val": self.current_val}
        print(f"##### info in reset: {info} #####")

        return self.current_obs, info

    def step(self, action):
        # Map the action to obs
        self.current_action = self._get_action(action, initial=False)
        self.current_obs = self._get_obs(self.current_action)
        reward = self._get_reward(self.current_obs)
        info = {"current_obs": self.current_obs, "current_action": self.current_action, "current_val": self.current_val}
        print(f"##### info in step: {info} #####")
        print(f"##### Reward: {reward} #####")

        self.termination_cnt += 1
        print(f"##### Termination_cnt: {self.termination_cnt} #####")
        terminated = False
        if self.termination_cnt == self.max_iter:
            terminated = True
        truncated = False

        return self.current_obs, reward, terminated, truncated, info

gym.register(
    id="SAR10B-v1",
    entry_point=SAR10BitEnv1,
)
