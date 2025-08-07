import pickle
import numpy as np
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.problems.dynamic.df import DF1
from tqdm import tqdm

problem = DF1(t=0)

allF = np.loadtxt("allFdf1.txt")
allX = np.loadtxt("allXdf1.txt")
resF = np.loadtxt("resFdf1.txt")

ideal_point = np.min(resF, axis=0)
nadir_point = np.max(resF, axis=0)
allF = (allF - ideal_point) / (nadir_point - ideal_point)
allF = np.clip(allF, 0, None)

begin_point = np.array([0.5] * allX.shape[1])
begin_F = problem.evaluate(begin_point.reshape(1, -1))[0]
begin_F = (begin_F - ideal_point) / (nadir_point - ideal_point)
begin_F = np.clip(begin_F, 0, None)
zero_point = np.zeros_like(begin_F)

front_indice = fast_non_dominated_sort(allF)
#print("len fronts {}".format(len(front_indice)))
#for i in range(len(front_indice)):
    ##print("len front {}: {}".format(i, len(front_indice[i])))

num_episodes = 10000
len_episodes = 20
episodes = []

for i in tqdm(range(num_episodes)):
    ##print()
    ##print(i)
    current_episode = []
    weights = np.random.rand(2)
    weights = weights / weights.sum()
    #print("weights", weights)
    goal_point = None
    diff = np.inf
    for temp_F in allF[front_indice[0]]:
        #F = np.clip(temp_F , goal_point, None)
        d1 = np.dot(temp_F - zero_point, weights) / np.sqrt(np.sum(weights ** 2))
        temp_point = zero_point + d1 * weights
        d2 = np.sqrt(np.sum((temp_F - temp_point) ** 2))
        temp_diff = d1 + 5 * d2
        if temp_diff < diff:
            diff = temp_diff
            goal_point = temp_F
    #print("goal_point", goal_point)
    #print("diff", diff)
    d_goal = np.sqrt(np.sum(goal_point ** 2))
    lambdas = goal_point / d_goal
    #print("d_goal", d_goal)
    #print("lambdas", lambdas)
    #print()
    start_point = np.hstack([begin_point, begin_F, goal_point])
    current_obs = start_point
    for j in range(len_episodes):
        index = len_episodes - j
        next_front = allF[front_indice[index]] 
        reward = 0
        next_index = front_indice[index][0]
        next_F = next_front[0]
        for i, temp_F in zip(front_indice[index], next_front):
            #F = np.clip(temp_F , goal_point, None)
            d0 = np.dot(temp_F - zero_point, lambdas)
            d1 = 0 if d0 < d_goal else d0 - d_goal
            temp_point = zero_point + d0 * lambdas
            d2 = np.sqrt(np.sum((temp_F - temp_point) ** 2))
            diff = d1 + 5 * d2
            temp_reward = 1 / (1 + diff ** 2)
            if temp_reward > reward:
                reward = temp_reward
                next_index = i
                next_F = temp_F
        #print("next_F", next_F)
        #print("d0: {} d1: {} d2: {}".format(d0, d1, d2))
        #print("reward", reward)
        next_x = allX[next_index]

        next_obs = np.hstack([next_x, next_F, goal_point])

        action = 0.5 * (next_x - current_obs[:len(next_x)]) + 0.5

        done = j == len_episodes - 1

        temp_dict = dict(
                            Observation=current_obs, 
                            Action=action, 
                            Next=next_obs, 
                            Reward=reward, 
                            Done=done
                        )
        current_episode.append([temp_dict["Observation"], temp_dict["Action"], temp_dict["Next"], temp_dict["Reward"]])

        current_obs = next_obs

    episodes.append(current_episode)

##print(np.array([[tup[0] for tup in episode] for episode in episodes]))
with open("df1_episodes.pkl", "wb") as f:
    pickle.dump(episodes, f)
