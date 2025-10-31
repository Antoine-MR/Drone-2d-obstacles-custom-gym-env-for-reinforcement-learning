import numpy as np
from typing import Tuple, Union


def compute_reward(
    env,
    action: np.ndarray, 
    terminated: bool,
    truncated: bool, 
    obs: np.ndarray
) -> float:
    velocity_x, velocity_y = obs[0], obs[1]
    angular_velocity = obs[2] 
    angle = obs[3]
    distance_x, distance_y = obs[4], obs[5]
    pos_x, pos_y = obs[6], obs[7]
    
    distance_reward = 1.0 / (np.abs(distance_x) + 0.1) + 1.0 / (np.abs(distance_y) + 0.1)
    stability_penalty = 0.1 * np.abs(angular_velocity) + 0.1 * np.abs(angle)
    velocity_penalty = 0.05 * (np.abs(velocity_x) + np.abs(velocity_y))
    action_penalty = 0.01 * (np.abs(action[0]) + np.abs(action[1]))
    terminal_penalty = 10.0 if terminated else 0.0
    
    total_reward = distance_reward - stability_penalty - velocity_penalty - action_penalty - terminal_penalty
    
    return total_reward