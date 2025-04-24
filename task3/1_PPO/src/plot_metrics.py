#!/usr/bin/env python
"""
Plots SB3 TensorBoard metrics and saves them as PNGs.

Exemple d'appel :
    python plot_metrics.py --log_dir logs/ppo_main --prefix main_run --fig_dir figures
"""

import argparse, glob, os, pathlib
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# ---------------------------------------------------------------------
# utilitaires
# ---------------------------------------------------------------------
def latest_event_file(log_dir: str, prefix: str) -> pathlib.Path:
    """Renvoie le chemin du .tfevents le plus récent pour un run donné."""
    runs = sorted(
        d for d in pathlib.Path(log_dir).iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    )
    if not runs:
        raise FileNotFoundError(f"Aucun sous-dossier « {prefix}* » dans {log_dir}")
    run_dir = runs[-1]                     # dernier run = plus récent
    ev_files = list(run_dir.glob("events.out.tfevents.*"))
    if not ev_files:
        raise FileNotFoundError(f"Aucun .tfevents dans {run_dir}")
    return ev_files[0]


def read_scalars(ev_file: pathlib.Path, tags):
    """Charge les scalaires demandés dans un dict {tag: (steps, values)}."""
    ea = event_accumulator.EventAccumulator(str(ev_file),
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    out = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in events]),
                        np.array([e.value for e in events]))
    return out


# ---------------------------------------------------------------------
# tracés
# ---------------------------------------------------------------------
def main(log_dir: str, prefix: str, fig_dir: str):
    ev_file = latest_event_file(log_dir, prefix)
    print(f"Lecture des logs : {ev_file}")

    tags_reward = ["rollout/ep_rew_mean", "rollout/ep_len_mean"]
    tags_loss   = ["train/loss", "train/value_loss", "train/entropy_loss"]

    rewards = read_scalars(ev_file, tags_reward)
    losses  = read_scalars(ev_file, tags_loss)

    fig_dir = pathlib.Path(fig_dir)
    fig_dir.mkdir(exist_ok=True, parents=True)

    # ---------- 1) Récompenses & longueurs d’épisode ----------
    plt.figure(figsize=(10, 4))
    for tag, (x, y) in rewards.items():
        plt.plot(x, y, label=tag.split('/')[-1])
    plt.xlabel("Timesteps")
    plt.title("Récompense et longueur moyenne par épisode")
    plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "rewards_lengths.png", dpi=150)

    # ---------- 2) Pertes ----------
    n = len(losses)
    fig, axs = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axs = [axs]
    for ax, (tag, (x, y)) in zip(axs, losses.items()):
        ax.plot(x, y, label=tag.split('/')[-1])
        ax.set_xlabel("Timesteps")
        ax.set_title(tag.split('/')[-1])
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "losses.png", dpi=150)

    print(f"Figures enregistrées dans  {fig_dir.resolve()}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/ppo_main",
                        help="Dossier racine contenant les runs TensorBoard")
    parser.add_argument("--prefix",  default="main_run",
                        help="Préfixe des sous-dossiers (main_run_1, …)")
    parser.add_argument("--fig_dir", default="figures",
                        help="Où stocker les .png générés")
    args = parser.parse_args()
    main(args.log_dir, args.prefix, args.fig_dir)
