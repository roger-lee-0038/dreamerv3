import random
import gymnasium as gym
import numpy as np
import torch
import rl_utils
from MultiDiscreteActionDDPG import MultiDiscreteActionDDPG
import pickle

num_episodes = 2000
init_num = 1
episode_length = 20 # 每条序列的最大长度
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

batch_size = 64
# update_interval = 100
buffer_size = 10000
minimal_size = 1000
# sigma = 0.01  # 高斯噪声标准差

np.bool8 = np.bool_

env_name = "envSAR10B:SAR10B-v0"
env = gym.make(env_name)
random.seed(1)
np.random.seed(1)
#env.seed(0)
env_seed = 1
env_options = {"init_num": init_num, "max_iter": episode_length}
torch.manual_seed(1)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
env.reset(seed = env_seed, options=env_options)
sub_state_dims = env.get_wrapper_attr("sub_observation_dims")
sub_action_dims = env.get_wrapper_attr("sub_action_dims")
agent = MultiDiscreteActionDDPG(init_num, sub_state_dims, sub_action_dims, hidden_dim, actor_lr, critic_lr, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent(env, env_seed, agent, num_episodes, replay_buffer, minimal_size, batch_size, env_options=env_options)

with open("trainSAR10B_results.pkl", "wb") as handler:
    pickle.dump((env_name, return_list, agent), handler)
