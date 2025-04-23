# Importations #
import os
import sys
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import make_env
import random

# main #

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

env = make_env()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        self.obs_size = obs_size
        super(Net, self).__init__()

        self.fc1 = nn.Linear(obs_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

    



class DQN(): 
    def __init__(self,
                environment,
                gamma,
                batch_size,
                buffer_capacity,
                update_target_every, 
                epsilon_start, 
                decrease_epsilon_factor, 
                epsilon_min,
                learning_rate,
                hidden_size,
                ):
        self.env = deepcopy(environment)

        self.gamma = gamma
        
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every
        
        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = decrease_epsilon_factor
        self.epsilon_min = epsilon_min
        
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        
        self.reset()
        
    def get_action(self, state, epsilon=None):
        """
            Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None: 
            epsilon = self.epsilon
            
        if np.random.rand() < epsilon: 
            return self.env.action_space.sample()
        else: 
            return np.argmax(self.get_q(state))
    
    def update(self, state, action, reward, terminated, next_state):
        self.buffer.push(
            torch.tensor(state).unsqueeze(0).to(device),
            torch.tensor([[action]], dtype=torch.int64).to(device),
            torch.tensor([reward], dtype=torch.float32).to(device),
            torch.tensor([terminated], dtype=torch.int64).to(device),
            torch.tensor(next_state).unsqueeze(0).to(device),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        transitions = self.buffer.sample(self.batch_size)
        state_batch, action_batch, reward_batch, terminated_batch, next_state_batch = [
            torch.cat(items).to(device) for items in zip(*transitions)
        ]

        # Q values
        values = self.q_net(state_batch).gather(1, action_batch)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)
            max_next_q = next_q.max(1)[0].view(-1, 1)
            targets = reward_batch.view(-1, 1) + (1 - terminated_batch.view(-1, 1)) * self.gamma * max_next_q

        # Loss and optimization
        loss = self.loss_function(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target net update
        if (self.n_steps + 1) % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()
        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.item()
    
    
    def get_q(self, state):
        """
        Compute Q function for a state
        """
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.q_net.forward(state_tensor)
        return output.to("cpu").numpy()[0] 
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
                        np.exp(-1. * self.n_eps / self.decrease_epsilon_factor ) )

    def reset(self):
        n1 = len(self.env.unwrapped.config["observation"]["features"])
        minx, maxx = self.env.unwrapped.config["observation"]["grid_size"][0]
        miny, maxy = self.env.unwrapped.config["observation"]["grid_size"][1]
        nx = (maxx-minx)//self.env.unwrapped.config["observation"]["grid_step"][0]
        ny = (maxy-miny)//self.env.unwrapped.config["observation"]["grid_step"][1]

        obs_size = n1*nx*ny

        n_actions = self.env.action_space.n 
        self.q_net =  Net(obs_size, self.hidden_size, n_actions)
        self.target_net = Net(obs_size, self.hidden_size, n_actions)

        self.q_net.to(device)
        self.target_net.to(device)

        
        self.buffer = ReplayBuffer(self.buffer_capacity)
        
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        
        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0
