import gymnasium as gym
import numpy as np
import pickle

# env = gym.make("envSAR10B:SAR10B-v0")
# print(env.observation_space)
# print(env.action_space)
# print(dir(env))
# print(env.get_wrapper_attr("resFcapdac"))
# print(env.get_wrapper_attr("sub_observation_dims"))
# print(env.get_wrapper_attr("sub_action_dims"))
# print(env.reset(seed=10, options={"init_num":5, "max_iter":100}))
# multihot_action = np.array(
#     [
#         [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
#         [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
#         [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], 
#         [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], 
#         [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], 
#      ])
# print(env.step(multihot_action))

def evaluate(env, agent, n_episode=5, episode_length=50):
    # 对学习的策略进行评估,此时不会进行探索
    returns = []
    for _ in range(n_episode):
        tot_rew = 0
        obs = env.reset(seed=1, options={"zdt_name": "zdt4", "init_num":5, "max_iter": episode_length})[0]
        for _ in range(episode_length):
            actions = agent.take_action(obs, explore=False)
            obs, rew, terminated, truncated, info = env.step(actions)
            tot_rew += np.sum(rew)
        returns.append(tot_rew) 
    return returns

env = gym.make("envZDT:ZDT-v0")
with open("trainZDT_results.pkl", "rb") as handler:
    _, _, agent = pickle.load(handler)

ev_returns = evaluate(env, agent, n_episode=5, episode_length=50)
print(f"ev_returns: {ev_returns}")