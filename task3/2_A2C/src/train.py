
import os, time, pickle, psutil, gymnasium as gym, highway_env      # noqa: F401
import matplotlib.pyplot as plt

from stable_baselines3 import A2C                  # <- CHANGE
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ------------------------------------------------------------------
#  1. Chemins
# ------------------------------------------------------------------
ROOT       = os.path.abspath("..")                    # /project/task3
CONFIG_FN  = os.path.join(ROOT, "configs", "config3.pkl")
LOG_DIR    = os.path.join(ROOT, "logs", "a2c_main")   # <- CHANGE
MODEL_DIR  = os.path.join(ROOT, "models")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Charger la configuration HighwayEnv
# ------------------------------------------------------------------
with open(CONFIG_FN, "rb") as f:
    cfg = pickle.load(f)

# ------------------------------------------------------------------
# 3. Création d’un environnement visuel pour un “sanity-check”
# ------------------------------------------------------------------
env_vis = gym.make("racetrack-v0", render_mode="rgb_array")
env_vis.unwrapped.configure(cfg)
obs, _ = env_vis.reset()
plt.imshow(env_vis.render()); plt.axis("off"); plt.title("Vue initiale"); plt.show()
env_vis.close()

# ------------------------------------------------------------------
# 4. Vecteur d’environnements pour A2C (Subproc)
# ------------------------------------------------------------------
CORES  = psutil.cpu_count(logical=False)      # ex. 24
N_ENVS = min(22, CORES - 2)                   # 22 envs max
print(f"{N_ENVS=} (CPU physiques : {CORES})")

vec_env = make_vec_env(
    "racetrack-v0",
    n_envs=N_ENVS,
    env_kwargs={"config": cfg},
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={"start_method": "fork"},
)

# ------------------------------------------------------------------
# 5. Hyper-paramètres A2C (diffèrent de PPO)
# ------------------------------------------------------------------
#  n_steps = nombre de transitions PAR ENV avant chaque update A2C
N_STEPS     = 8                      # 8 * N_ENVS ≈ 176 transitions / update
TOTAL_TS    = 300_000                # par env  → ≈ 6 M transitions totales
LR          = 7e-4                   # lr par défaut A2C
GAMMA       = 0.99
ENT_COEF    = 0.0                    # entropy bonus
VF_COEF     = 0.5

policy_kwargs = dict(net_arch=[dict(pi=[256, 256],
                                    vf=[256, 256])])

model = A2C(
    "MlpPolicy",
    vec_env,
    n_steps=N_STEPS,
    learning_rate=LR,
    gamma=GAMMA,
    ent_coef=ENT_COEF,
    vf_coef=VF_COEF,
    policy_kwargs=policy_kwargs,
    tensorboard_log=LOG_DIR,
    verbose=1,
)

# ------------------------------------------------------------------
# 6. Callback d’évaluation périodique
# ------------------------------------------------------------------
eval_env = gym.make("racetrack-v0")
eval_env.unwrapped.configure(cfg)

eval_cb = EvalCallback(
    eval_env,
    eval_freq=25_000,         # un peu plus rare (A2C update + rapide)
    n_eval_episodes=5,
    log_path=LOG_DIR,
    deterministic=True,
)

# ------------------------------------------------------------------
# 7. Entraînement
# ------------------------------------------------------------------
start = time.time()
model.learn(
    total_timesteps=TOTAL_TS,
    tb_log_name="main_run",
    callback=eval_cb,
    progress_bar=True,
)
print("⏱  Durée entraînement :", (time.time() - start) / 60, "min")

# ------------------------------------------------------------------
# 8. Sauvegarde
# ------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "a2c_racetrack_fast.zip")
model.save(model_path)
with open(model_path.replace(".zip", "_info.txt"), "w") as f:
    f.write(
        f"algo=A2C\nn_envs={N_ENVS}\n"
        f"n_steps={N_STEPS}\n"
        f"lr={LR}\ngamma={GAMMA}\n"
        f"total_ts/env={TOTAL_TS}\n"
    )
print("✓ modèle & méta sauvegardés →", model_path)
