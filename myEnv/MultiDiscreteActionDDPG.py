import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rl_utils import *

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MultiDiscreteActionDDPG:
    ''' DDPG with discrete actions'''
    def __init__(self, sub_num, sub_state_dims, sub_action_dims, hidden_dim, actor_lr, critic_lr, tau, gamma, device):

        self.sub_num = sub_num

        # Given a sub-state, each sub-actor net serves for selecting an sum(action_dims)-D sub_action, including len(action_dims) onehot vectors 
        self.sub_action_dims = sub_action_dims
        self.state_dim = self.sub_num * len(sub_state_dims)
        self.sub_action_dim = sum(sub_action_dims)
        self.sub_actors = []
        self.target_sub_actors = []
        self.sub_actor_optimizers = []
        for _ in range(sub_num):
            self.sub_actors.append(TwoLayerFC(self.state_dim, self.sub_action_dim, hidden_dim).to(device))
            self.target_sub_actors.append(TwoLayerFC(self.state_dim, self.sub_action_dim, hidden_dim).to(device))
        for target_sub_actor, sub_actor in zip(self.target_sub_actors, self.sub_actors):
            target_sub_actor.load_state_dict(sub_actor.state_dict())
            self.sub_actor_optimizers.append(torch.optim.Adam(sub_actor.parameters(), lr=actor_lr))

        # Initialize sub-critic nets
        self.critic_input_dim = self.state_dim + self.sub_num * self.sub_action_dim
        self.sub_critics = []
        self.target_sub_critics = []
        self.sub_critic_optimizers = []
        for _ in range(sub_num):
            self.sub_critics.append(TwoLayerFC(self.critic_input_dim, 1, hidden_dim).to(device))
            self.target_sub_critics.append(TwoLayerFC(self.critic_input_dim, 1, hidden_dim).to(device))
        for target_sub_critic, sub_critic in zip(self.target_sub_critics, self.sub_critics):
            target_sub_critic.load_state_dict(sub_critic.state_dict())
            self.sub_critic_optimizers.append(torch.optim.Adam(sub_critic.parameters(), lr=critic_lr))

        self.tau = tau
        self.gamma = gamma
        self.device = device

    def get_sub_multihot(self, sub_action, explore=True, eps=0.01):
        sub_multihot = []
        start_dim = 0
        for i in range(len(self.sub_action_dims)):
            if explore:
                onehot = gumbel_softmax(sub_action[:,start_dim:start_dim+self.sub_action_dims[i]])
            else:
                onehot = onehot_from_logits(sub_action[:,start_dim:start_dim+self.sub_action_dims[i]], eps=eps)
            start_dim += self.sub_action_dims[i]
            sub_multihot.append(onehot)
        return torch.cat(sub_multihot, dim=1)

    def take_action(self, state, explore=True):
        state = torch.tensor(state, dtype=torch.float).view(-1, self.state_dim).to(self.device)
        print(f"state: {state}")
        assert state.shape[0] == 1
        multihot_action = []
        for sub_actor in self.sub_actors:
            sub_action = sub_actor(state)
            print(f"sub_action: {sub_action}")
            # 1 * sum(self.sub_action_dims)
            sub_multihot = self.get_sub_multihot(sub_action, explore=explore, eps=0.01)
            print(f"sub_multihot: {sub_multihot}")
            multihot_action.append(sub_multihot)
        # self.sub_num * sum(self.sub_action_dims)
        multihot_action = torch.cat(multihot_action, dim=0)
        print(f"multihot_action: {multihot_action}")
        return multihot_action.detach().cpu().numpy()

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        print(f"##### Info of trandition_dict #####")
        print(f"##### states: {transition_dict['states'][0].shape} #####")
        print(f"##### actions: {transition_dict['actions'][0].shape} #####")
        print(f"##### rewards: {transition_dict['rewards'][0].shape} #####")
        print(f"##### next_states: {transition_dict['next_states'][0].shape} #####")
        states = torch.tensor(transition_dict['states'], dtype=torch.float).view(-1,self.state_dim).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,self.sub_num*self.sub_action_dim).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, self.sub_num).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).view(-1,self.state_dim).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Centralized training for sub-critic nets, if the elements of rewards are the same, then all sub-critic nets will be the same
        all_target_sub_act = [
            self.get_sub_multihot(pi(next_states), explore=False) for pi in self.target_sub_actors
        ]
        target_sub_critic_input = torch.cat((next_states, *all_target_sub_act), dim=1)
        # next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        for target_sub_critic, sub_critic, reward, sub_critic_optimizer in zip(self.target_sub_critics, self.sub_critics, rewards.T, self.sub_critic_optimizers):
            next_q_values = target_sub_critic(target_sub_critic_input)
            print(f"next_q_values: {next_q_values.shape}")
            print(f"reward: {reward.shape}")
            q_targets = reward.unsqueeze(dim=-1) + self.gamma * next_q_values * (1 - dones)
            sub_critic_input = torch.cat((states, actions), dim=1)
            critic_loss = torch.mean(F.mse_loss(sub_critic(sub_critic_input), q_targets))
            sub_critic_optimizer.zero_grad()
            critic_loss.backward()
            sub_critic_optimizer.step()

        # Training for each sub-actor net
        for i_sub_actor, (sub_actor, sub_actor_optimizer) in enumerate(zip(self.sub_actors, self.sub_actor_optimizers)):
            cur_sub_actor_out = sub_actor(states)
            cur_act_vf_in = self.get_sub_multihot(cur_sub_actor_out, explore=True)
            all_sub_actor_acs = []
            for i, pi in enumerate(self.sub_actors):
                if i == i_sub_actor:
                    all_sub_actor_acs.append(cur_act_vf_in)
                else:
                    all_sub_actor_acs.append(self.get_sub_multihot(pi(states), explore=False))
            vf_in = torch.cat((states, *all_sub_actor_acs), dim=1)
            sub_actor_loss = -self.sub_critics[i_sub_actor](vf_in).mean()
            sub_actor_loss += (cur_sub_actor_out**2).mean() * 1e-3
            sub_actor_optimizer.zero_grad()
            sub_actor_loss.backward()
            sub_actor_optimizer.step()

        for sub_actor, target_sub_actor in zip(self.sub_actors, self.target_sub_actors):
            self.soft_update(sub_actor, target_sub_actor)  # 软更新策略网络
        for sub_critic, target_sub_critic in zip(self.sub_critics, self.target_sub_critics):
            self.soft_update(sub_critic, target_sub_critic)  # 软更新价值网络