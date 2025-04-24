#!/usr/bin/env python
"""
Évalue le modèle PPO sauvegardé, imprime reward moyenne & longueur.
Usage :
python evaluate_agent.py --episodes 10 --model ../models/sac_racetrack_fast.zip
"""

import argparse, pickle, numpy as np, gymnasium as gym, highway_env     # noqa: F401
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="../models/sac_racetrack_fast.zip")
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()

# Charger config
with open("../configs/config3.pkl", "rb") as f:
    cfg = pickle.load(f)

# Charger modèle
model = PPO.load(args.model, env=None)
env = gym.make("racetrack-v0")
env.unwrapped.configure(cfg)

rewards, lengths = [], []
for ep in range(args.episodes):
    obs,_ = env.reset()
    done = truncated = False; tot = step = 0
    while not (done or truncated):
        act,_ = model.predict(obs, deterministic=True)
        obs,r,done,truncated,_ = env.step(act)
        tot += r; step += 1
    rewards.append(tot); lengths.append(step)
    print(f"Épisode {ep+1}/{args.episodes} → R={tot:.1f}  L={step}")

env.close()
print(f"\nReward moyenne : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
print(f"Longueur moy.  : {np.mean(lengths):.0f}  ± {np.std(lengths):.0f}")
