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


def compute_reward(env, action, terminated, truncated, obs):
    velocity_x, velocity_y = obs[0], obs[1]
    angular_velocity = obs[2] 
    angle = obs[3]
    distance_x, distance_y = obs[4], obs[5]
    pos_x, pos_y = obs[6], obs[7]
    
    distance_reward = 1.0 / (np.abs(distance_x) + 0.1) + 1.0 / (np.abs(distance_y) + 0.1)
    terminal_penalty = 10.0 if terminated else 0.0
    
    return distance_reward - terminal_penalty