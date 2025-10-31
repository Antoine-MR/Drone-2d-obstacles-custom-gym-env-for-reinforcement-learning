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
def compute_reward(self, action, terminated: bool, truncated: bool, obs):
    """
    This function computes the reward for the drone environment.
    
    :param action: The action taken by the agent.
    :param terminated: Whether the episode has been terminated.
    :param truncated: Whether the episode has been truncated (optional).
    :param obs: The observation of the drone at this step, including 
                position, velocity, target_distance, and angular_velocity.

    :return A tuple containing whether the episode was truncated and terminated, 
            respectively, and the reward for this step.
    """
    
    # Extract relevant variables from the observation
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)
    
    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(obs['angle'][0]))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty 
              + angle_reward + boundary_penalty + terminal_penalty)

    return truncated, terminated, float(reward)
