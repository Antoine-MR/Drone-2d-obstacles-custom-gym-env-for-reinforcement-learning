from typing import Callable, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rl_zoo3.wrappers import MaskVelocityWrapper

try:
    import pybullet_envs_gymnasium
except ImportError:
    pass

try:
    import highway_env
except ImportError:
    pass
else:
    # hotfix for highway_env
    import numpy as np

    np.float = np.float32  # type: ignore[attr-defined]

try:
    import custom_envs
except ImportError:
    pass

try:
    import gym_donkeycar
except ImportError:
    pass

try:
    import panda_gym
except ImportError:
    pass

try:
    import rocket_lander_gym
except ImportError:
    pass

try:
    import minigrid
except ImportError:
    pass

# Import drone environment for Eureka
try:
    import sys
    import os
    drone_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'drone_2d_custom_gym_env_package')
    sys.path.insert(0, drone_path)
    import drone_2d_custom_gym_env
    from eureka.envs.drone_2d import Drone2DWrapper
    
    # Register the Drone2D environment
    register(
        id='Drone2D-v0',
        entry_point='eureka.envs.drone_2d:Drone2DWrapper',
        max_episode_steps=1000,
    )
except ImportError as e:
    print(f"Warning: Could not import drone environment: {e}")
    pass


# Register no vel envs
def create_no_vel_env(env_id: str) -> Callable[[Optional[str]], gym.Env]:
    def make_env(render_mode: Optional[str] = None) -> gym.Env:
        env = gym.make(env_id, render_mode=render_mode)
        env = MaskVelocityWrapper(env)
        return env

    return make_env


for env_id in MaskVelocityWrapper.velocity_indices.keys():
    name, version = env_id.split("-v")
    register(
        id=f"{name}NoVel-v{version}",
        entry_point=create_no_vel_env(env_id),  # type: ignore[arg-type]
    )
