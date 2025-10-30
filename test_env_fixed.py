"""
Test rapide de l'environnement avec obstacles et nouvelles observations
"""
import os
import sys

# Ajouter le répertoire du package à sys.path
package_dir = os.path.join(os.path.dirname(__file__), 'drone_2d_custom_gym_env_package')
sys.path.insert(0, package_dir)

import numpy as np
from drone_2d_custom_gym_env import Drone2dEnv

def test_environment():
    print("🚁 Test de l'environnement avec obstacles...")
    
    # Test avec obstacles
    print("\n1. Test avec obstacles:")
    env_with_obs = Drone2dEnv(render_sim=False, use_obstacles=True, num_obstacles=3)
    obs = env_with_obs.reset()[0]
    print(f"   Taille observation: {obs.shape}")
    print(f"   Observation: {obs}")
    
    # Test une action
    action = env_with_obs.action_space.sample()  # Action valide
    obs, reward, done, truncated, info = env_with_obs.step(action)
    print(f"   Après une action {action}: reward={reward}, done={done}")
    print(f"   Nouvelle observation: {obs}")
    
    # Test sans obstacles
    print("\n2. Test sans obstacles:")
    env_no_obs = Drone2dEnv(render_sim=False, use_obstacles=False, num_obstacles=0)
    obs = env_no_obs.reset()[0]
    print(f"   Taille observation: {obs.shape}")
    print(f"   Observation: {obs}")
    
    # Test des détecteurs d'obstacles
    print("\n3. Test détection obstacles:")
    env_with_obs.reset()
    if hasattr(env_with_obs, 'get_obstacle_distances'):
        distances = env_with_obs.get_obstacle_distances()
        print(f"   Distances obstacles: {distances}")
    
    print("\n✅ Tests réussis!")

if __name__ == "__main__":
    test_environment()