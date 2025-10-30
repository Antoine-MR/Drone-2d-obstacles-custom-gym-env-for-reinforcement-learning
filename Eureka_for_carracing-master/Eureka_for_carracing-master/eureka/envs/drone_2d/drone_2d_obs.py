import numpy as np
import sys
import os

# Add drone package path
drone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'drone_2d_custom_gym_env_package')
sys.path.insert(0, drone_path)

import drone_2d_custom_gym_env
from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
import gymnasium as gym
from gymnasium import spaces


class Drone2DCustom(Drone2dEnv):
    def __init__(self, render_sim=False, render_path=False, render_shade=False, shade_distance=70,
                 n_steps=1000, n_fall_steps=10, change_target=False, initial_throw=True):
        super().__init__(render_sim=render_sim, render_path=render_path, render_shade=render_shade, 
                         shade_distance=shade_distance, n_steps=n_steps, n_fall_steps=n_fall_steps,
                         change_target=change_target, initial_throw=initial_throw)

    def step(self, action):
        obs, _, done, info = super().step(action)
        
        terminated = done
        truncated = False
        step_reward = compute_reward(self, action, terminated, truncated, obs)

        return obs, step_reward, terminated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)  # type: ignore
        obs, info = super().reset()
        return obs


def create_env():
    return Drone2DCustom(
        render_sim=False,
        render_path=False,
        render_shade=False,
        n_steps=1000,
        n_fall_steps=10,
        change_target=False,
        initial_throw=True
    )


def compute_reward(self, action, truncated: bool, terminated: bool, obs):
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
    velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
    angular_velocity_penalty = -1.0 * abs(angular_velocity) / max_angular_velocity

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    obstacle_distance_x, obstacle_distance_y = obs['obstacle_distance'][0], obs['obstacle_distance'][1]
    max_obstacle_distance = 5.0
    if np.abs(obstacle_distance_x) > max_obstacle_distance or np.abs(obstacle_distance_y) > max_obstacle_distance:
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
    velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
    angular_velocity_penalty = -1.0 * abs(angular_velocity) / max_angular_velocity

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    obstacle_distance_x, obstacle_distance_y = obs['obstacle_distance'][0], obs['obstacle_distance'][1]
    max_obstacle_distance = 5.0
    if np.abs(obstacle_distance_x) > max_obstacle_distance or np.abs(obstacle_distance_y) > max_obstacle_distance:
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
    velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
    angular_velocity_penalty = -1.0 * abs(angular_velocity) / max_angular_velocity

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    obstacle_distance_x, obstacle_distance_y = obs['obstacle_distance'][0], obs['obstacle_distance'][1]
    max_obstacle_distance = 5.0
    if np.abs(obstacle_distance_x) > max_obstacle_distance or np.abs(obstacle_distance_y) > max_obstacle_distance:
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
    velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / max_velocity
    angular_velocity_penalty = -1.0 * abs(angular_velocity) / max_angular_velocity

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    obstacle_distance_x, obstacle_distance_y = obs['obstacle_distance'][0], obs['obstacle_distance'][1]
    max_obstacle_distance = 5.0
    if np.abs(obstacle_distance_x) > max_obstacle_distance or np.abs(obstacle_distance_y) > max_obstacle_distance:
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

    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity']
    angle = obs['angle']
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for exceeding maximum velocity or angular velocity
    velocity_penalty = -1.0 * max(np.abs(velocity_x), np.abs(velocity_y)) / 100.0
    angular_velocity_penalty = -1.0 * abs(angular_velocity) / 2.0

    # Reward for maintaining stability (angle close to zero)
    angle_reward = 10.0 * (1.0 - abs(angle) / 180.0)

    # Avoid obstacles (distance between drone and obstacle is greater than a certain threshold)
    distance_x, distance_y = obs['obstacle_distance'][0], obs['obstacle_distance'][1]
    if np.abs(distance_x) > 5.0 or np.abs(distance_y) > 5.0:
        obstacle_reward = -10.0
    else:
        obstacle_reward = 0.0

    # Combine rewards with appropriate weights
    reward = (target_reward + angle_reward + velocity_penalty + angular_velocity_penalty +
              obstacle_reward)

    return truncated, terminated, float(reward)
