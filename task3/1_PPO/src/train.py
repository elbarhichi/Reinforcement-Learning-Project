#!/usr/bin/env python
"""
Entraîne PPO sur HighwayEnv « racetrack-v0 », avec multiprocessing SubprocVecEnv,
TensorBoard et sauvegarde modèle + config.

Par défaut :
  - 80 % des cœurs physiques disponibles
  - 5e5 pas d’entraînement
  - dossiers : logs/ppo_main  et  models/ppo_main
"""

import argparse, datetime, os, pickle, psutil, time, pathlib
import gymnasium as gym
import highway_env                     # noqa – enregistre les envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


# ----------------------------------------------------------------------
# arguments CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--env_id",     default="racetrack-v0",
                   help="ID Gymnasium de l’environnement")
    p.add_argument("--config",     default="configs/config3.pkl",
                   help="Pickle contenant le dict de configuration HighwayEnv")
    p.add_argument("--timesteps",  type=float, default=5e5,
                   help="Nombre de pas d’entraînement")
    p.add_argument("--n_envs",     type=int,   default=None,
                   help="Nbr d’env. simultanés (par défaut : 80 %% des cœurs)")
    p.add_argument("--log_root",   default="logs/ppo_main",
                   help="Dossier racine TensorBoard")
    p.add_argument("--model_root", default="models/ppo_main",
                   help="Dossier où sauver .zip et .pkl")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Graine aléatoire (répétabilité)")
    return p.parse_args()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def next_run_id(log_root: pathlib.Path, prefix="main_run") -> str:
    """Renvoie un identifiant de run incrémental : main_run_1, _2, …"""
    existing = sorted(d for d in log_root.glob(f"{prefix}_*") if d.is_dir())
    return f"{prefix}_{len(existing)+1}"


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # charge la config highway
    with open(args.config, "rb") as f:
        cfg = pickle.load(f)

    # nombre d’envs parallèles
    if args.n_envs is None:
        cpu_phys = psutil.cpu_count(logical=False) or 1
        args.n_envs = max(1, int(cpu_phys * .8))     # 80 % des cœurs
    print(f" → n_envs = {args.n_envs}  |  total_timesteps = {int(args.timesteps):,}")

    # dossiers sortie
    log_root   = pathlib.Path(args.log_root)
    model_root = pathlib.Path(args.model_root)
    run_id     = next_run_id(log_root)
    tb_dir     = log_root / run_id
    model_root.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # crée l’environnement vectorisé
    vec_env = make_vec_env(
        args.env_id,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": cfg},
        seed=args.seed,
    )

    # définit l’agent PPO
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024 // args.n_envs,          # taille batch compatible
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        target_kl=0.03,
        verbose=1,
        tensorboard_log=str(tb_dir),
        seed=args.seed,
    )

    # ---------------------------------------------------------------
    print(f"\n[START] Entraînement PPO – logs : {tb_dir}")
    t0 = time.time()
    model.learn(total_timesteps=int(args.timesteps), progress_bar=True)
    dt = time.time() - t0
    print(f"[DONE]  entraînement terminé en {dt/60:.1f} min")

    # ---------------------------------------------------------------
    # sauvegarde
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mdl_path  = model_root / f"ppo_racetrack_{timestamp}.zip"
    cfg_path  = model_root / f"config_{timestamp}.pkl"

    model.save(mdl_path)
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)

    print(f"Modèle : {mdl_path}")
    print(f"Config  : {cfg_path}")

    # nettoie proprement
    vec_env.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
