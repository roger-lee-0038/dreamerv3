from typing import Optional
import numpy as np
import gymnasium as gym
import math
from pymoo.problems.dynamic.df import DF1


class DF1Env2(gym.Env):

    def __init__(self, max_iter=100, index=0):
        self.problem = DF1(t=0)

        self.pareto_fronts = None
        self.ideal_point = None
        self.nadir_point = None
        self.start_point = None
        self.goal_point = None
        self.start_val = np.inf
        self.current_val = np.inf
        
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
            low = np.array([0.0] * self.problem.n_var + [-1e30] * self.n_objs),
            high = np.array([1.0] * self.problem.n_var + [1e30] * self.n_objs),
            shape = (self.problem.n_var + self.n_objs,),
            dtype = np.float32,
        )

        self.action_space = gym.spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (self.problem.n_var,),
            dtype = np.float32,
        )

    def _get_obs(self, action):
        self.current_obs[:self.problem.n_var] = np.clip(
            self.current_obs[:self.problem.n_var] + 2 * action - 1,
            0,
            1,
        )
        return self.current_obs

    def _get_reward(self, obs):
        F = self.problem.evaluate(self.current_obs[:self.problem.n_var].reshape(1, -1))
        #diff = np.clip((obs[:, :self.n_objs] - self.goal_point), 0, None)
        diff = F[0] - self.goal_point
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
        nor_weights = self.weights * np.array([1, 1])

        #self.pareto_fronts = np.loadtxt("/export/home/liwangzhen/Research/dreamerv3/for_df1/resFdf1.txt")
        self.pareto_fronts = np.loadtxt("/export/home/liwangzhen/Research/dreamerv3/for_df1/front_51.txt")
        #self.pareto_sets = np.loadtxt("/Users/Roger/Desktop/dreamerv3/pymooTestSAR/for_df1/resXdf1.txt")

        self.start_point = np.array([0.5] * self.problem.n_var)
        goal_index = np.argmin(np.tensordot(self.pareto_fronts, nor_weights, axes=([-1], [-1])))
        self.goal_point = self.pareto_fronts[goal_index]

        self.current_obs = np.hstack([self.start_point, self.goal_point])
        self.current_val = self._get_reward(self.current_obs)
        self.start_val = self._get_reward(self.current_obs)

        info = {"goal": self.goal_point, "start": self.start_point, "start reward": self.start_val, "current obs": self.current_obs[:self.problem.n_var], "current reward": self.current_val}
        print(f"##### info in reset: \n{info}")

        return self.current_obs, info

    def step(self, action):
        # Map the action to obs
        self.current_obs = self._get_obs(action)
        reward = self._get_reward(self.current_obs)
        info = {"goal": self.goal_point, "start": self.start_point, "start reward": self.start_val, "current obs": self.current_obs[:self.problem.n_var], "current reward": self.current_val}
        print(f"##### info in step: \n{info}")

        self.termination_cnt += 1
        print(f"##### Termination_cnt: {self.termination_cnt}")
        terminated = False
        if self.termination_cnt == self.max_iter:
            terminated = True
        truncated = False

        return self.current_obs, reward, terminated, truncated, info

gym.register(
    id="DF1-v2",
    entry_point=DF1Env2,
)
