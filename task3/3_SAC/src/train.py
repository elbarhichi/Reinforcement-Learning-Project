
import os, time, pickle, psutil, gymnasium as gym, highway_env     # noqa: F401
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from Racetracks.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# ------------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------------
ROOT       = os.path.abspath("..")                     # /project/task3
CONFIG_FN  = os.path.join(ROOT, "configs", "config3.pkl")
LOG_DIR    = os.path.join(ROOT, "logs", "sac_main")    # <- dossier spécifique
MODEL_DIR  = os.path.join(ROOT, "models")
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Charger la config Highway
# ------------------------------------------------------------------
with open(CONFIG_FN, "rb") as f:
    cfg = pickle.load(f)

# ------------------------------------------------------------------
# 3. Sanity-check visuel (facultatif)
# ------------------------------------------------------------------
env_vis = gym.make("racetrack-v0", render_mode="rgb_array")
env_vis.unwrapped.configure(cfg)
plt.imshow(env_vis.render()); plt.axis("off"); plt.title("Vue initiale"); plt.show()
env_vis.close()

# ------------------------------------------------------------------
# 4. Vecteur d’environnements
#    SAC est off-policy → 1 seul env suffit (sample replay buffer)
#    mais on peut quand même en lancer plusieurs (Subproc)  
# ------------------------------------------------------------------
CORES  = psutil.cpu_count(logical=False)
N_ENVS = min(8, CORES//2)          # 1 à 8 envs, selon ta machine
print(f"{N_ENVS=} (CPU physiques : {CORES})")

vec_env = make_vec_env(
    "racetrack-v0",
    n_envs=N_ENVS,
    env_kwargs={"config": cfg},
    vec_env_cls=SubprocVecEnv if N_ENVS > 1 else None,
    vec_env_kwargs={"start_method": "fork"} if N_ENVS > 1 else None,
)

# ------------------------------------------------------------------
# 5. Hyper-paramètres SAC
# ------------------------------------------------------------------
BUFFER_SIZE     = 1_000_000      # replay buffer
BATCH_SIZE      = 256
LEARNING_STARTS = 10_000         # steps avant 1er update
TRAIN_FREQ      = 1              # update chaque step
GRADIENT_STEPS  = 1              # n° updates / train_freq
GAMMA           = 0.99
TAU             = 0.005          # soft update target networks
LR              = 3e-4

policy_kwargs = dict(net_arch=[256, 256])

model = SAC(
    "MlpPolicy",
    vec_env,
    buffer_size=BUFFER_SIZE,
    learning_rate=LR,
    batch_size=BATCH_SIZE,
    train_freq=TRAIN_FREQ,
    gradient_steps=GRADIENT_STEPS,
    learning_starts=LEARNING_STARTS,
    gamma=GAMMA,
    tau=TAU,
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
    eval_freq=50_000,         # SAC = updates quasi continus → eval moins fréquente
    n_eval_episodes=5,
    log_path=LOG_DIR,
    deterministic=True,
)

# ------------------------------------------------------------------
# 7. Entraînement
# ------------------------------------------------------------------
TOTAL_TS = 1_000_000           # steps par ENV (≈1–5 M conseillé pour SAC)
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
model_path = os.path.join(MODEL_DIR, "sac_racetrack.zip")
model.save(model_path)
with open(model_path.replace(".zip", "_info.txt"), "w") as f:
    f.write(
        f"algo=SAC\nn_envs={N_ENVS}\n"
        f"buffer_size={BUFFER_SIZE}\nbatch_size={BATCH_SIZE}\n"
        f"lr={LR}\ngamma={GAMMA}\n"
        f"total_ts/env={TOTAL_TS}\n"
    )
print("✓ modèle & méta sauvegardés →", model_path)
