# Importations #
import os
from copy import deepcopy
from datetime import datetime
from itertools import product
from pprint import pprint

import numpy as np
import torch
from torch.multiprocessing import freeze_support, set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass

from dqn import DQN
from dqn import env as dqn_env



def run_n_episodes(agent, env, display=False, save_videos=None):
    """
    Joue quelques épisodes avec l'agent, peut afficher ou enregistrer en vidéo.
    """
    import imageio
    all_rewards = []
    frames = []

    for i in range(3):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_frames = []

        while not done:
            if display or save_videos:
                frame = env.render()
                episode_frames.append(frame)

            action = agent.get_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        all_rewards.append(total_reward)

        if save_videos:
            os.makedirs(save_videos, exist_ok=True)
            path = os.path.join(save_videos, f"episode_{i+1}.gif")
            imageio.mimsave(path, episode_frames, fps=5)
            frames.append(path)

    return all_rewards, frames

def current_date_str():
    now = datetime.now()
    return f"{now.year:04d}-{now.month:02d}-{now.day:02d}"


def train(env, agent: DQN, N_episodes: int, run_name: str):
    """
    Entraîne l'agent sur un certain nombre d'épisodes, log et sauvegarde le meilleur modèle.
    """
    total_time = 0
    losses = []
    episode_rewards = []
    best_reward = -np.inf
    date_str = current_date_str()
    full_run_name = f"{run_name}_{date_str}"

    log_dir = os.path.join("logs", full_run_name)
    os.makedirs(log_dir, exist_ok=True)

    state, _ = env.reset()

    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            loss_val = agent.update(state, action, reward, terminated, next_state)

            state = next_state
            episode_loss += loss_val
            episode_reward += reward
            done = terminated or truncated
            total_time += 1

        losses.append(episode_loss)
        episode_rewards.append(episode_reward)

        print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.4f}")

        with open(os.path.join(log_dir, "training_metrics.csv"), "a") as f:
            f.write(f"{ep+1},{episode_reward},{episode_loss},{agent.epsilon}\n")

        if (ep+1) % 10 == 0:
            video_dir = os.path.join(log_dir, f"episode_{ep+1}")
            eval_rewards, frames = run_n_episodes(agent, env, display=False, save_videos=video_dir)
            mean_reward = np.mean(eval_rewards)
            print(f"[Evaluation] Mean Reward after {ep+1} episodes: {mean_reward:.2f}")

            with open(os.path.join(log_dir, "eval_metrics.csv"), "a") as f:
                f.write(f"{ep+1},{mean_reward}\n")

            if mean_reward > best_reward:
                best_reward = mean_reward
                print("[Saved] New best model with reward:", best_reward)
                model_path = os.path.join(log_dir, "best_model.pt")
                torch.save(agent.q_net.state_dict(), model_path)

    return losses

def run_network_train(batch_size, learning_rate, hidden_size, epsilon_start, decrease_epsilon_factor, epsilon_min, N_episodes, gamma):
    """
    Lance un entraînement avec les hyperparamètres donnés.
    """
    run_env = deepcopy(dqn_env)

    agent = DQN(
        run_env,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=10_000,
        update_target_every=32,
        epsilon_start=epsilon_start,
        decrease_epsilon_factor=decrease_epsilon_factor,
        epsilon_min=epsilon_min,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
    )

    params = {
        "epsilon_start": epsilon_start,
        "epsilon_decrease_factor": decrease_epsilon_factor,
        "epsilon_min": epsilon_min,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "N_episodes": N_episodes,
        "hidden_size": hidden_size,
        "gamma": gamma,
    }

    pprint(params)
    train(run_env, agent, N_episodes, run_name="DQN_Experiment")

def parameter_search(
    batch_sizes=[128],
    learning_rates=[1e-3],
    hidden_sizes=[16],
    epsilon_starts=[0.95],
    decrease_epsilon_factors=[1000],
    epsilon_mins=[0.05],
    gammas=[0.99],
    N_episodes=100,
):
    """
    Teste plusieurs configs d'hyperparamètres via une recherche en grille.
    """
    parameter_grid = product(
        batch_sizes,
        learning_rates,
        hidden_sizes,
        epsilon_starts,
        decrease_epsilon_factors,
        epsilon_mins,
        [N_episodes],
        gammas,
    )

    for params in parameter_grid:
        run_network_train(*params)

if __name__ == "__main__":
    freeze_support()
    parameter_search(

        batch_sizes=[128],
        learning_rates=[1e-3],
        hidden_sizes=[64], 
        epsilon_starts=[0.95],
        decrease_epsilon_factors=[1000],
        epsilon_mins=[0.05],
        N_episodes=5000, 
        gammas=[0.99],
    )
