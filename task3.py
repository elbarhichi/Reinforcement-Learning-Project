import pickle

ENV= "parking-v0"
config_dict = {
    "observation": {
        "type": "Kinematics",  # Type d'observation donnant accès aux données brutes (position, vitesse, etc.)
        "vehicles_count": 1,   # Pour le parking, on se concentre souvent sur la seule voiture contrôlée
        "features": ["x", "y", "vx", "vy", "heading"],
        "normalize": True,     # Normalisation des observations pour faciliter l'apprentissage
        "absolute": True,      # Positions absolues utiles pour une tâche de parking
    },
    "action": {
        "type": "ContinuousAction",  # Actions continues pour un contrôle précis (accélération et direction)
        "longitudinal": True,
        "lateral": True,
    },
    "duration": 120,  # Durée de la simulation en secondes (suffisante pour un parking)
    "controlled_vehicles": 1,
    "other_vehicles": 0,  # Pas de trafic dans un scénario de parking minimaliste (peut être ajouté selon vos envies)
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 6,     # Ajustement de l'échelle pour une meilleure visualisation
    "render_agent": True,
    "show_trajectories": True,
    "offscreen_rendering": False,
    "simulation_frequency": 15,  # Fréquence de simulation
    "policy_frequency": 5,       # Fréquence d'action (mise à jour de la politique)
    # Récompenses adaptées pour la tâche de parking :
    "collision_reward": -1,         # Pénalité en cas de collision
    "parking_success_reward": 5,    # Récompense lorsque le véhicule se gare correctement
    "distance_to_goal_reward": 0.1,   # Récompense progressive en fonction de la proximité de la place de parking
}

with open("configs/config3.pkl", "wb") as f:
    pickle.dump(config_dict, f)
