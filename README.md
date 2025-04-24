# Reinforcement Learning Project

## Team Members

- El Barhichi Mohammed
- Bouhadida Malek
- Ammar Mariem

---


## Environments and Scenarios

### 1. Task 1: Pre-specified Environment : **Highway with Discrete Actions**


### 2. Task 2: **Racetracks with Continuous Actions**


### 3. Task 3: **Racetracks Environment using StableBaselines3**


```bash
pip install -r requirements.txt

# 1. Entraîner (~200K pas / 22 envs)
python src/train_task3.py --timesteps 1000000 --n_envs 22

# 2. Tracer la courbe reward
python src/plot_metrics.py --logdir logs/racetrack_ppo_fast

# 3. Évaluer le modèle
python src/evaluate_agent.py --model models/ppo_racetrack_fast.zip --episodes 10
```


### 3. Task 4: **Racetracks Environment**


