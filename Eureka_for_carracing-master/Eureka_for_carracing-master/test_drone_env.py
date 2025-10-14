import sys
import os
from eureka.envs.drone_2d.drone_2d_env import Drone2DWrapper

# Add drone package path
drone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'drone_2d_custom_gym_env_package'))
sys.path.insert(0, drone_path)

try:
    env = Drone2DWrapper()
    print('Drone environment wrapper created successfully')
    
    obs, info = env.reset()
    print(f'Reset successful. Observation shape: {obs.shape}')
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step successful. Reward: {reward:.4f}')
    
    print("Drone environment test passed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        if 'env' in locals():
            env.close()
    except:
        pass