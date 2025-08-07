from typing import Optional
import numpy as np
import gymnasium as gym
from pymoo.problems.dynamic.df import DF1
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
import os


class DF1Env4(gym.Env):

    def __init__(self, max_iter=100, index=0):
        self.problem = DF1(t=0)

        self.pareto_fronts = None
        self.ideal_point = None
        self.nadir_point = None
        self.start_point = None
        self.goal_point = None
        self.zero_point = None
        self.d1 = None
        self.d2 = None
        self.d_goal = None
        self.current_val = None
        self.F = None
        self.start_val = None
        self.start_F = None
        
        # The number of objectives
        self.n_objs = 2
        self.n_mets = 2
        # The maximal number of iterations
        self.max_iter = max_iter
        # The counter for maximal number of iterations
        self.termination_cnt = 0

        self.current_obs = None
        self.weights = np.random.rand(self.n_objs)
        self.weights = self.weights / self.weights.sum()

        self.observation_space = gym.spaces.Box(
            low = -1e30, 
            high = 1e30,
            shape = (self.problem.n_var + 2 * self.n_objs,),
            dtype = np.float32,
        )

        self.action_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (self.problem.n_var,),
            dtype = np.float32,
        )

    def get_addFile(self):
        return self.addFile

    def get_n_objs(self):
        return self.n_objs

    def _get_obs(self, action):
        temp_action = np.clip(
            self.current_obs[self.n_objs:self.n_objs + self.problem.n_var] + 2 * action - 1,
            0,
            1,
        )
        self.F = self.problem.evaluate(temp_action.reshape(1, -1))[0]
        self.F = (self.F - self.ideal_point) / (self.nadir_point - self.ideal_point)
        #self.F = np.clip(self.F, 0, None)
        self.current_obs = np.hstack([self.F, temp_action, self.goal_point]).astype(np.float32)
        return self.current_obs

    def _get_reward(self, obs):
        F = np.clip(self.F, self.zero_point, None)
        d0 = np.dot(F - self.zero_point, self.weights) / np.sqrt(np.sum(self.weights ** 2))
        if d0 < self.d_goal:
            #self.d1 = 0 if d0 < self.d_goal else d0 - self.d_goal
            self.d1 = 0
            temp_point = self.zero_point + d0 * self.weights
            self.d2 = np.sqrt(np.sum((F - temp_point) ** 2))
            #diff = self.d1 + 5 * self.d2
            diff = self.d2
        else:
            self.d1 = d0 - self.d_goal
            temp_point = self.zero_point + d0 * self.weights
            self.d2 = np.sqrt(np.sum((F - temp_point) ** 2))
            #diff = np.sqrt(self.d1 ** 2 + self.d2 ** 2)
            diff = self.d1 + self.d2

        reward = 1 / (1 + np.sum(diff ** 2))
        self.current_val = reward
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
        self.pareto_file = "/Users/Roger/Desktop/dreamerv3/for_df1/resFdf1.txt"
        self.pareto_fronts = np.loadtxt(self.pareto_file)
        self.ideal_point = np.min(self.pareto_fronts, axis=0)
        self.nadir_point = np.max(self.pareto_fronts, axis=0)

        #self.pareto_fronts = np.loadtxt("/export/home/liwangzhen/Research/dreamerv3/for_df1/front_51.txt")
        self.pareto_fronts = (self.pareto_fronts - self.ideal_point) / (self.nadir_point - self.ideal_point)

        self.addFile = "/Users/Roger/Desktop/dreamerv3/for_df1/df1v4.txt"

        if os.path.exists(self.addFile):
            temp_fronts = np.loadtxt(self.addFile)
            if len(temp_fronts):
                self.pareto_fronts = np.vstack([self.pareto_fronts, temp_fronts])
        #self.zero_point = np.zeros(self.n_objs)
        self.zero_point = np.min(self.pareto_fronts, axis=0) # not strictly zero

        self.start_point = np.array([0.5] * self.problem.n_var)
        self.F = self.problem.evaluate(self.start_point.reshape(1, -1))[0]
        self.F = (self.F - self.ideal_point) / (self.nadir_point - self.ideal_point)
        #self.F = np.clip(self.F, 0, None)
        #goal_index = np.argmin(np.tensordot(self.pareto_fronts, self.weights, axes=([-1], [-1])))
        #ref_d = np.sqrt(np.sum(self.pareto_fronts[ref_index] ** 2))
        #self.goal_point = self.weights / np.sqrt(np.sum(self.weights ** 2)) * ref_d * 0.5
        diff = np.inf
        goal_index = 0
        for i, temp_F in enumerate(self.pareto_fronts):
            d0 = np.dot(temp_F - self.zero_point, self.weights) / np.sqrt(np.sum(self.weights ** 2))
            d1 = d0
            temp_point = self.zero_point + d1 * self.weights
            d2 = np.sqrt(np.sum((temp_F - temp_point) ** 2))
            temp_diff = d1 + 5 * d2
            if temp_diff < diff:
                diff = temp_diff
                goal_index = i

        #self.goal_point = self.pareto_fronts[goal_index]
        self.goal_point = 0.5 * self.pareto_fronts[goal_index] + 0.5 * self.zero_point
        self.d_goal = np.sqrt(np.sum((self.goal_point - self.zero_point) ** 2))
        self.weights = self.goal_point / self.d_goal

        self.current_obs = np.hstack([self.F, self.start_point, self.goal_point]).astype(np.float32)
        self.current_val = self._get_reward(self.current_obs)

        self.start_val = self.current_val
        self.start_F = self.F

        info = {"weight": self.weights, "goal": self.goal_point, "start F": self.start_F, "d1": self.d1, "d2": self.d2, "start reward": self.start_val} 
        print(f"##### info in reset: \n{info}")

        return self.current_obs, info

    def step(self, action):
        # Map the action to obs
        self.current_obs = self._get_obs(action)
        reward = self._get_reward(self.current_obs)
        info = {"weight": self.weights, "goal": self.goal_point, "start F": self.start_F, "start reward": self.start_val, "current obs": self.current_obs[:self.problem.n_var], "current F": self.F, "d1": self.d1, "d2": self.d2, "current reward": self.current_val}

        if not np.any(np.all(self.pareto_fronts <= self.F, axis=1) & np.any(self.pareto_fronts < self.F, axis=1)):
            if not np.any(np.all(self.pareto_fronts == self.F, axis=1)):
                self.pareto_fronts = np.vstack([self.pareto_fronts, self.F])
                info.update({"add_F": True})
        print(f"##### info in step: \n{info}")


        self.termination_cnt += 1
        print(f"##### Termination_cnt: {self.termination_cnt}")
        terminated = False
        if self.termination_cnt == self.max_iter:
            terminated = True
        truncated = False

        return self.current_obs, reward, terminated, truncated, info

gym.register(
    id="DF1-v4",
    entry_point=DF1Env4,
)
