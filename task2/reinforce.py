# Imports
import torch
import torch.optim as optim
import numpy as np

from net import Net, NetContinousActions


class REINFORCEContinuous:
    """Implementation of the REINFORCE algorithm for continuous action spaces."""
    
    def __init__(self, action_space, observation_space, gamma, episode_batch_size, lr, net_hidden_size=128):
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.episode_batch_size = episode_batch_size
        self.lr = lr
        self.hidden_size = net_hidden_size

        self.reset()
        
        
    def reset(self):
        """Reset the agent's policy."""
        
        obs_size = int(np.prod(self.observation_space.shape))
        actions_dim = self.action_space.shape[0]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        self.policy_net = NetContinousActions(obs_size, self.hidden_size, actions_dim).to(self.device)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        
        self.scores = []
        self.current_episode = []
        self.n_eps = 0


    def update(self, state, action, reward, terminated, next_state):
        """Update the policy."""
        
        self.current_episode.append((torch.tensor(state, device=self.device).unsqueeze(0),
                                     torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0),
                                     torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0),
                                    ))

        if terminated:
            self.n_eps += 1
            states, actions, rewards = tuple([torch.cat(data).to(self.device) for data in zip(*self.current_episode)])
            current_episode_returns = self._returns(rewards, self.gamma)
            std = current_episode_returns.std()
            
            # we normalize to reduce variance
            if std > 1e-6:
                current_episode_returns = (current_episode_returns - current_episode_returns.mean()) / std
            else:
                current_episode_returns = current_episode_returns - current_episode_returns.mean()

            means, stds = self.policy_net.forward(states)
            dist = torch.distributions.Normal(means, stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            score = log_probs * current_episode_returns 
            self.scores.append(score.sum().unsqueeze(0))
            
            self.current_episode = []
            
            # we learn by updating the policy
            if (self.n_eps % self.episode_batch_size) == 0:
                
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
                self.optimizer.step()

                self.scores = []

                return full_neg_score.item()
            
        return None


    def _returns(self, rewards, gamma):
        """Compute the discounted returns."""
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32, device=self.device)


    def get_action(self, state, epsilon=None):
        """Sample an action from the policy."""
        
        state_tensor = (torch.as_tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0))
        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
            action = torch.distributions.Normal(mean, std).sample()
            low = torch.tensor(self.action_space.low, device=self.device)
            high = torch.tensor(self.action_space.high, device=self.device)
            return action.clamp(low, high).cpu().numpy()[0]


    def train_reset(self):
        """Reset the training."""
        
        self.scores = []
        self.current_episode = []

