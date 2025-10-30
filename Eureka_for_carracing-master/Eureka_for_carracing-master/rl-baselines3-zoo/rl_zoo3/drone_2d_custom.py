import numpy as np
from typing import Tuple, Union


def compute_reward(
    env,
    action: np.ndarray, 
    terminated: bool,
    truncated: bool, 
    obs: np.ndarray
) -> float:
        return get_reward(observation, terminated)
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
def get_reward(observation, terminated):
    """
    Computes the reward for controlling a 2D drone.

    Args:
        observation (dict): The current state of the drone.
            It should contain the following keys:
                - position (list): [x, y] coordinates of the drone's position.
                - velocity (list): [x, y] components of the drone's velocity.
                - angular_velocity (float): The drone's angular velocity around its vertical axis.
                - target_position (list): [x, y] coordinates of the target position.
                - obstacles (list): List of [x, y] positions of nearby obstacles.

    Returns:
        reward (float): The computed reward value between [-1, 1].
    """

    # Compute distance to target
    target_reward = 0.5 / (observation['position'][0]**2 + observation['position'][1]**2) ** 0.5

    # Penalty for high velocity
    velocity_penalty = -0.01 * (observation['velocity'][0]**2 + observation['velocity'][1]**2) ** 0.5

    # Reward for maintaining stable angle
    angle_reward = 1.0 * (1.0 - abs(observation['angular_velocity']))

    # Penalize for being out of bounds or close to obstacles
    boundary_penalty = 0.0
    if observation['position'][0] < 0.1 or observation['position'][0] > 10:
        boundary_penalty -= 10.0
    elif observation['position'][1] < 0.1 or observation['position'][1] > 10:
        boundary_penalty -= 10.0
    for obstacle in observation['obstacles']:
        distance_to_obstacle = ((observation['position'][0] - obstacle[0])**2 + (observation['position'][1] - obstacle[1])**2) ** 0.5
        if distance_to_obstacle < 1:
            boundary_penalty -= 10.0

    # Terminal penalty for ending the episode
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards and normalize to [-1, 1] range
    reward = (target_reward + velocity_penalty + angle_reward + boundary_penalty + terminal_penalty) / 5

    return max(min(reward, 1), -1)
