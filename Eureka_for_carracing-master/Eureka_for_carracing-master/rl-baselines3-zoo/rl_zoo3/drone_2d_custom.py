import numpy as np
from typing import Tuple, Union


def compute_reward(
    env,
    action: np.ndarray, 
    terminated: bool,
    truncated: bool, 
    obs: np.ndarray
) -> float:
        return compute_reward(self, action, terminated, truncated, obs)
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
def compute_reward(self, action: np.ndarray, terminated: bool, truncated: bool, obs):
    """
    Reward function for reaching a target position while avoiding obstacles and maintaining stability.
    
    :param action: The action taken by the agent (not used in this reward function)
    :param terminated: Whether the episode has ended
    :param truncated: Whether the episode was terminated prematurely
    :param obs: The current observation of the drone's state
    :return: A tuple containing the updated termination and truncation flags, and the computed reward
    """

    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 10.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.05 * (np.abs(velocity_x) + np.abs(velocity_y))

    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 5.0 * (1.0 - np.abs(angle / np.pi))  # Normalize angle to [-pi, pi] range

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -20.0

    # Reward for avoiding obstacles (heuristic approach)
    obstacle_reward = 10.0 * (1.0 / (np.linalg.norm(obs['obstacle_distance']) + 0.5))

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + obstacle_reward)

    return terminated, truncated, float(reward)
