# Reinforcement Learning Project

## Team Members

- El Barhichi Mohammed
- Bouhadida Malek
- Ammar Mariem

---


## Environments and Scenarios

### 1. Task 1: Pre-specified Environment : **Highway with Discrete Actions**


### 2. Task 2: **Racetracks with Continuous Actions**


### 3. Task 3 : **Racetrack – Comparaison PPO / A2C / SAC (en utilisant SB3)**

Nous avons évalué trois familles d’acteur-critic sur `racetrack-v0`
(observation *OccupancyGrid*, contrôle latéral continu) :

| Algo | Pas totaux | n envs | Durée ≈ | Mean Reward (10 ep) |
|------|-----------:|-------:|--------:|--------------------:|
| PPO  | 100 k      | 4      | 15 min  | **138 ± 28**        |
| A2C  | 90 k       | 4      | 12 min  | 360 ± 55 \*         |
| SAC  | 60 k       | 4      | 10 min  |  92 ± 31 (early)    |

\* forte variance, ~20 % de collisions.

Scripts (dossier `task3/`) :

```bash
# 0. dépendances
pip install -r requirements.txt

# 1. Entraînement rapide (ex. PPO 100k pas – 4 environnements)
python task3/train_ppo.py      --timesteps 100000   --n_envs 4

#    Variante A2C
python task3/train_a2c.py      --timesteps  90000   --n_envs 4

#    Variante SAC (budget réduit)
python task3/train_sac_fast.py --timesteps  60000   --n_envs 4

# 2. Visualiser les métriques TensorBoard / PNG
python task3/plot_metrics.py   --logdir logs/ppo_main_run_1
open figures/ppo_training_curves.png        # (mac) ou xdg-open …

# 3. Évaluer un modèle sauvé
python task3/evaluate_agent.py \
       --model models/ppo_racetrack_fast.zip --episodes 10

# 4. Visualiser la dernière trajectoire
python task3/plot_trajectory.py \
       --model models/ppo_racetrack_fast.zip --env racetrack-v0
```


### 4. Task 4: **Racetracks Environment**


