from typing import Optional
import numpy as np
import gymnasium as gym
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.util.dominator import Dominator
from pymoo.indicators.hv import Hypervolume
import math
from pymoo.problems import get_problem


class ZDTEnv(gym.Env):

    def __init__(self, zdt_name="zdt1", init_num=1, max_iter=20):
        self.problem = get_problem(zdt_name)
        # The observation dimensionalities
        self.sub_observation_dims = [10] * self.problem.n_var
        # The action dimensionalities
        self.sub_action_dims = [3] * self.problem.n_var
        # The number of objectives
        self.n_objs = self.problem.n_obj

        # The number of initial points
        self.init_num = init_num
        # The maximal number of iterations
        self.max_iter = max_iter
        # The counter for maximal number of iterations
        self.termination_cnt = 0

        self.current_obs = None
        self.current_behaviour = None

        # Observations are indices in the 6*6 CAPDAC matrix and 3*3 COM matrix
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array(self.init_num * [self.sub_observation_dims])+1,
            seed = 1,
        )

        # The actions are movements along the indices of observations, with 0, 1, 2 indicating 1-step down, no movement and 1-step up
        self.action_space = gym.spaces.MultiDiscrete(
            np.array(self.init_num * [self.sub_action_dims]),
            seed = 1,
        )

    # def _get_obs_and_info(self):
    #     return self.current_obs, self.current_behaviour

    def _get_action(self, multihot_action):
        action = []
        for sub_multihot_action in multihot_action:
            sub_action = []
            start_dim = 0
            for i in range(len(self.sub_action_dims)):
                onehot = sub_multihot_action[start_dim: start_dim+self.sub_action_dims[i]]
                sub_action.append(np.where(onehot==1)[0][0])
                start_dim += self.sub_action_dims[i]
            print(f"#### Get sub-action: {sub_action}")
            action.append(sub_action)
        return np.array(action)
    
    def _get_behaviour(self, obs):
        print("#### Enter _get_behaviour #####")
        behaviour = []
        for indices in obs:
            print(f"#### Indices: {indices} #####")
            sub_behaviour = self.problem.evaluate(indices/10)
            print(f"#### Get sub_behaviour: {sub_behaviour} #####")
            behaviour.append(sub_behaviour)
        return np.array(behaviour)

    def _normalize(self, input, ideal_point, nadir_point):
        print(f"input: {input}")
        print(f"ideal_point: {ideal_point}")
        print(f"nadir_point: {nadir_point}")
        mask = (ideal_point == nadir_point)[0]
        output = np.ones_like(input)
        output[:,~mask] = (input[:,~mask] - ideal_point[:,~mask]) / (nadir_point[:,~mask] - ideal_point[:,~mask])
        return output

    def _get_reward(self, new_behaviour, old_behaviour):
        ideal_point = np.min(np.vstack([new_behaviour, old_behaviour]), axis=0, keepdims=True)
        nadir_point = np.max(np.vstack([new_behaviour, old_behaviour]), axis=0, keepdims=True)
        new_vecs = self._normalize(new_behaviour, ideal_point, nadir_point)
        old_vecs = self._normalize(old_behaviour, ideal_point, nadir_point)

        new_areas = np.prod(1.1-new_vecs, axis=1)
        old_areas = np.prod(1.1-old_vecs, axis=1)
        individual_rewards = new_areas - old_areas
        print(f"##### Individual rewards: {individual_rewards} #####")

        if self.init_num > 1:
            metric = Hypervolume(
                ref_point=np.array([1.1]*self.n_objs),
                norm_ref_point=False,
                zero_to_one=False,
                ideal=np.zeros(self.n_objs),
                nadir=np.ones(self.n_objs),
                )
            hv_new = metric.do(new_vecs)
            hv_old = metric.do(old_vecs)
            shared_rewards = hv_new - hv_old
            print(f"##### Shared rewards: {shared_rewards} #####")
        else:
            shared_rewards = 0

        rewards = 100 * (individual_rewards + shared_rewards)
        return rewards

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if options.get("zdt_name"):
            zdt_name = options.get("zdt_name")
            self.problem = get_problem(zdt_name)
            # The observation dimensionalities
            self.sub_observation_dims = [5] * self.problem.n_var
            # The action dimensionalities
            self.sub_action_dims = [3] * self.problem.n_var
            # The number of objectives
            self.n_objs = self.problem.n_obj
            self.observation_space = gym.spaces.MultiDiscrete(
                np.array(self.init_num * [self.sub_observation_dims]),
                seed = 1,
            )
            self.action_space = gym.spaces.MultiDiscrete(
                np.array(self.init_num * [self.sub_action_dims]),
                seed = 1,
            )
        if options.get("init_num"):
            init_num = options.get("init_num")
            self.init_num = init_num
            self.observation_space = gym.spaces.MultiDiscrete(
                np.array(self.init_num * [self.sub_observation_dims]),
                seed = 1,
            )
            self.action_space = gym.spaces.MultiDiscrete(
                np.array(self.init_num * [self.sub_action_dims]),
                seed = 1,
            )
        if seed is not None:
            self.observation_space.seed(seed)
            self.action_space.seed(seed)
        if options.get("max_iter"):
            self.max_iter = options.get("max_iter")
        self.termination_cnt = 0

        # Initialize
        self.current_obs = self.observation_space.sample()
        self.current_behaviour = self._get_behaviour(self.current_obs)

        return self.current_obs, {"current_behaviour": self.current_behaviour}

    def step(self, multihot_action):
        print(f"##### Input multi-hot action for step {multihot_action} #####")
        action = self._get_action(multihot_action)
        print(f"##### Input movement action for step {action} #####")

        # Map the action to behaviour 
        self.current_obs = np.clip(self.current_obs + action - 1, 0, self.init_num * [self.sub_observation_dims])
        temp_behaviour = self._get_behaviour(self.current_obs)

        reward = self._get_reward(temp_behaviour, self.current_behaviour)
        print(f"##### Reward: {reward} #####")
        self.current_behaviour = temp_behaviour

        self.termination_cnt += 1
        print(f"##### Termination_cnt: {self.termination_cnt} #####")
        print(f"##### Max_iter: {self.max_iter} #####")
        terminated = False
        if self.termination_cnt == self.max_iter:
            terminated = True
        truncated = False

        return self.current_obs, reward, terminated, truncated, {"current_behaviour": self.current_behaviour}

gym.register(
    id="ZDT-v0",
    entry_point=ZDTEnv,
)