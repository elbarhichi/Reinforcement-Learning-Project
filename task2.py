import pickle

ENV = "highway-fast-v0"
config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction"
    },
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 60,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1, 
    "right_lane_reward": 0.5, 
    "high_speed_reward": 0.1, 
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  
    "simulation_frequency": 5,
    "policy_frequency": 1, 
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  
    "screen_height": 150, 
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

with open("configs/config2.pkl", "wb") as f:
    pickle.dump(config_dict, f)

# import gymnasium as gym
# import highway_env

# env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.unwrapped.configure(config)
# print(env.reset())