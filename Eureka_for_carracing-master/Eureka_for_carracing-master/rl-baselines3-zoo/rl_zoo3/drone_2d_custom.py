from typing import Tuple, Union

import torch
import numpy as np
import sys
import os
from pymunk import Vec2d

# Ajouter le chemin vers le package du drone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env
from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
import gymnasium as gym


class Drone2DCustom(Drone2dEnv):
    def __init__(self, render_sim=False, render_path=False, render_shade=False, shade_distance=70,
                 n_steps=500, n_fall_steps=10, change_target=False, initial_throw=True):
        # Désactiver le rendu pour l'entraînement Eureka (plus rapide)
        super().__init__(render_sim=render_sim, render_path=render_path, render_shade=render_shade, 
                         shade_distance=shade_distance, n_steps=n_steps, n_fall_steps=n_fall_steps,
                         change_target=change_target, initial_throw=initial_throw)

    def step(self, action):
        # Call parent step method to get the original behavior
        obs, _, done, info = super().step(action)
        
        # Replace the original reward with Eureka's optimized reward
        terminated = done
        truncated = False  # L'environnement original ne distingue pas terminated/truncated
        step_reward = self.get_reward(action, terminated, truncated, obs)

        return obs, step_reward, terminated, info

    def get_reward(self, action, terminated, truncated, obs) -> float:
        """Reward function that will be optimized by Eureka"""
        return compute_reward(self, action, terminated, truncated, obs)


def compute_reward(
        self,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        obs: np.ndarray,
) -> float:
    """Computes reward for Drone2D environment."""
    
    # Extract observation components
    velocity_x, velocity_y = obs[0], obs[1]
    angular_velocity = obs[2]
    angle = obs[3]
    distance_x, distance_y = obs[4], obs[5]
    pos_x, pos_y = obs[6], obs[7]
    
    # Base reward - closer to target is better
    distance_reward = 1.0 / (np.abs(distance_x) + 0.1) + 1.0 / (np.abs(distance_y) + 0.1)
    
    # Stability reward - penalize high angular velocity and extreme angles
    stability_reward = -0.1 * np.abs(angular_velocity) - 0.1 * np.abs(angle)
    
    # Velocity control reward - penalize excessive speeds
    velocity_penalty = -0.05 * (np.abs(velocity_x) + np.abs(velocity_y))
    
    # Action smoothness reward - penalize large actions
    action_penalty = -0.01 * (np.abs(action[0]) + np.abs(action[1]))
    
    # Terminal penalty
    if terminated:
        return -10.0
    
    # Combine rewards
    total_reward = distance_reward + stability_reward + velocity_penalty + action_penalty
    
    return total_reward