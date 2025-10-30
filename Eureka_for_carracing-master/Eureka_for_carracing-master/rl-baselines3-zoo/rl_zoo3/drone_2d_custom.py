import numpy as np
from typing import Tuple, Union


def compute_reward(
    env,
    action: np.ndarray, 
    terminated: bool,
    truncated: bool, 
    obs: np.ndarray
) -> float:
        return compute_reward(self, action, truncated, terminated, obs)
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
def compute_reward(self, action: Union[np.ndarray, int], truncated: bool, terminated: bool, obs) -> Tuple[bool, bool, float]:
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity']
    angle = obs['angle']
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for exceeding maximum velocity or angular velocity
    max_velocity = 50.0
    max_angular_velocity = 10.0
    if action == "stop":
        velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
        angular_velocity_penalty = -1.0 * abs(angular_velocity) / max_angular_velocity
    else:
        velocity_penalty = 1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
        angular_velocity_penalty = 1.0 * abs(angular_velocity) / max_angular_velocity

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    max_obstacle_distance = 5.0
    if np.abs(obs['obstacle_distance'][0]) > max_obstacle_distance or np.abs(obs['obstacle_distance'][1]) > max_obstacle_distance:
        obstacle_reward = -10.0
    else:
        obstacle_reward = 0.0

    # Penalize for going outside the boundaries
    boundary_penalty = 0.0
    if np.abs(pos_x) > 100.0 or np.abs(pos_y) > 100.0:
        boundary_penalty = -5.0

    # Penalize for falling down
    height_penalty = 0.0
    initial_height = obs['initial_position'][1]
    current_height = pos_y
    if current_height < initial_height:
        height_penalty = -10.0

    # Combine rewards with appropriate weights
    reward = (target_reward + angle_reward + velocity_penalty + angular_velocity_penalty +
              obstacle_reward + boundary_penalty + height_penalty)

    return truncated, terminated, float(reward)
