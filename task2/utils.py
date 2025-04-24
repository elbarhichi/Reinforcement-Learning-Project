import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv



def make_vectorised_env(env_name, config, n_envs):
    """Create a vectorized environment for parallel training."""
    
    def make_env(config):
        return gym.make(env_name, config=config, render_mode="rgb_array")
    
    return AsyncVectorEnv([lambda config=config: make_env(config) for _ in range(n_envs)])


def eval_agent(agent, env, n_sim=10):
    """Sequential version of Monte Carlo evaluation of the agent"""
    
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        
        while not done:
            action = agent.get_action(state, epsilon=0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
        episode_rewards[i] = reward_sum
        
    return episode_rewards


def eval_agent_vector(agent, vec_env, n_sim=25):
    """Parallelized version of Monte Carlo evaluation of the agent"""
    
    obs, _ = vec_env.reset()
    rewards = np.zeros(n_sim)
    dones = np.zeros(n_sim, dtype=bool)

    while not np.all(dones):
        actions = np.array([agent.get_action(o, epsilon=0) for o in obs])
        obs, reward, terminated, truncated, _ = vec_env.step(actions)
        dones |= terminated | truncated
        rewards += reward * (~dones)

    return rewards


def get_smoothed_returns(rewards_list, smooth_over=5):
    """Smooth the rewards list using a moving average."""
    
    if len(rewards_list) < smooth_over:
        return np.array(rewards_list)
    smoothed = np.convolve(rewards_list, np.ones(smooth_over)/smooth_over, mode='valid')
    
    return smoothed

