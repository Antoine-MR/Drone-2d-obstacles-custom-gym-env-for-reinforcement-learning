import numpy as np
import sys
import os

# Add drone package path
drone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'drone_2d_custom_gym_env_package')
sys.path.insert(0, drone_path)

try:
    import drone_2d_custom_gym_env
    from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
except ImportError as e:
    print(f"Error importing drone environment: {e}")
    print(f"Make sure the drone package is installed and accessible at: {drone_path}")
    raise


def create_drone_2d_env():
    """Factory function to create the Drone2D environment for Eureka."""
    return Drone2dEnv(
        render_sim=False,      # Disable rendering for faster training
        render_path=False,     # Disable path rendering
        render_shade=False,    # Disable shade rendering  
        n_steps=1000,          # Longer episodes for better learning
        n_fall_steps=10,       # Initial stabilization steps
        change_target=False,   # Fixed target position
        initial_throw=True     # Random initial conditions
    )


class Drone2DWrapper:
    def __init__(self):
        self.env = create_drone_2d_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)  # type: ignore
        obs, info = self.env.reset()
        return obs, info
        
    def step(self, action):
        result = self.env.step(action)  # type: ignore
        obs, _, terminated, truncated, info = result  # type: ignore
        
        reward = compute_reward_drone_2d(self.env, action, terminated, obs)
        
        return obs, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        return self.env.render(mode)
        
    def close(self):
        return self.env.close()


def compute_reward_drone_2d(env, action, terminated, obs):
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