import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """Basic neural net Implementation for the discrete setting."""

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
                            nn.Linear(obs_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, n_actions),
                        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class NetContinousActions(nn.Module):
    """Basic neural net Implementation for the continuous setting."""

    def __init__(self, obs_dim, hidden_size, act_dim):
        super().__init__()
        self.net = nn.Sequential(
                            nn.Linear(obs_dim, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, act_dim),
                        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  

    def forward(self, obs):
        obs = obs.view(obs.size(0), -1)
        mean = self.net(obs)
        std = torch.clamp(self.log_std, -20, 2).exp() 
        return mean, std
