import sys
import os

# Add drone package path
drone_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'drone_2d_custom_gym_env_package'))
sys.path.insert(0, drone_path)

try:
    import drone_2d_custom_gym_env
    from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv

    env = Drone2dEnv(render_sim=False)
    obs, info = env.reset()
    
    action = env.action_space.sample()
    result = env.step(action)
    print(f'Step returns {len(result)} values')
    
    if len(result) == 4:
        obs, reward, done, info = result
        print('Old Gym format: obs, reward, done, info')
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        print('New Gymnasium format: obs, reward, terminated, truncated, info')
    
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