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
        obs, _, done, truncated, info = super().step(action)
        
        terminated = done
        step_reward = compute_reward(self, action, terminated, truncated, obs)

        return obs, step_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)  # type: ignore
        obs, info = super().reset()
        return obs, info


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


def compute_reward(self, action, terminated: bool, truncated: bool, obs):
    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)

    """
    Fonction de recompense de base compatible avec le format dict
    Cette fonction sera remplacee par Eureka pendant l'entrainement
    """
    velocity_x, velocity_y = obs['velocity'][0], obs['velocity'][1]
    angular_velocity = obs['angular_velocity'][0]
    angle = obs['angle'][0]
    distance_x, distance_y = obs['target_distance'][0], obs['target_distance'][1]
    pos_x, pos_y = obs['position'][0], obs['position'][1]

    # Reward for reaching the target position
    target_reward = 2.0 / (np.abs(distance_x) + np.abs(distance_y) + 0.5)

    # Penalty for high velocity
    velocity_penalty = -0.01 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Penalty for high angular velocity
    angular_penalty = -0.01 * np.abs(angular_velocity)

    # Reward for maintaining stable angle (close to zero)
    angle_reward = 1.0 * (1.0 - np.abs(angle))

    # Penalize for being out of bounds
    boundary_penalty = 0.0
    if np.abs(pos_x) >= 0.9 or np.abs(pos_y) >= 0.9:
        boundary_penalty = -10.0

    # Terminal penalty
    terminal_penalty = -10.0 if terminated else 0.0

    # Combine rewards
    reward = (target_reward + velocity_penalty + angular_penalty + 
              angle_reward + boundary_penalty + terminal_penalty)

    return float(reward)
