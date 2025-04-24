# Reinforcement Learning Project

## Team Members

- El Barhichi Mohammed
- Bouhadida Malek
- Ammar Mariem

---


## Environments and Scenarios

### 1. Task 1: Pre-specified Environment : **Highway with Discrete Actions**
Dans cette tâche, l'objectif est d'implémenter un agent autonome utilisant **Deep Q-Networks (DQN)** pour naviguer sur une autoroute simulée. L'agent doit apprendre à :

- **Éviter les collisions** : En prenant des actions pour rester à une distance sécuritaire des autres véhicules.
- **Maintenir une vitesse optimale** : L'agent doit ajuster sa vitesse pour atteindre un comportement de conduite sécuritaire et efficace.
- **Changer de voie** : L'agent doit savoir quand changer de voie pour éviter les obstacles ou optimiser son parcours.

### Fichiers principaux

- **dqn.py** : Implémentation du réseau de neurones (DQN) qui génère des Q-values pour chaque action possible.
- **train_dqn.py** : Script principal pour entraîner l'agent en utilisant DQN.
- **config.py** : Paramètres de configuration pour le modèle, y compris les hyperparamètres comme le taux d'apprentissage et le facteur de réduction.

### Résultats et Fichiers d'Évaluation

Les résultats de l'entraînement et de l'évaluation de l'agent sont sauvegardés dans le dossier **dqn_evaluation_results**, qui contient les fichiers suivants :

- **best_model.pt** : Modèle sauvegardé après l'entraînement. Ce fichier contient les poids du réseau de neurones entraîné et peut être utilisé pour des tests ou évaluations supplémentaires.
- **eval_metrics.csv** : Contient les métriques d'évaluation de l'agent sur des épisodes non vus pendant l'entraînement. Cela permet de suivre la performance de l'agent pendant l'évaluation.
- **training_metrics.csv** : Contient les métriques d'entraînement, telles que les récompenses cumulées et la perte, utilisées pour suivre la progression du modèle au fil des épisodes.

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


