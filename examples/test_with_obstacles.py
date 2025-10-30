import gymnasium as gym
import sys
import os

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

print("Test de l'environnement avec obstacles")
print("Appuyez sur Ctrl+C pour arrêter")

# Test avec visualisation et obstacles activés
env = gym.make('drone-2d-custom-v0', 
               render_sim=True, render_path=True, render_shade=True,
               shade_distance=70, n_steps=500, n_fall_steps=10, 
               change_target=True, initial_throw=True,
               use_obstacles=True, num_obstacles=3)

obs, info = env.reset()
print("Environnement initialisé avec obstacles")

episode_count = 0
step_count = 0

try:
    while True:
        # Actions aléatoires pour tester (remplacez par votre agent entraîné)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        step_count += 1
        
        # Afficher les informations quand le drone percute quelque chose
        if done and step_count < 500:  # Si terminé avant la fin normale
            if abs(obs[6]) == 1 or abs(obs[7]) == 1:  # Sortie des limites
                print(f"Épisode {episode_count + 1} - Étape {step_count}: Drone sorti des limites")
            else:  # Collision avec obstacle
                print(f"Épisode {episode_count + 1} - Étape {step_count}: Collision avec obstacle!")
        elif done:  # Temps écoulé
            print(f"Épisode {episode_count + 1} - Étape {step_count}: Temps écoulé")
        
        if done:
            episode_count += 1
            step_count = 0
            obs, info = env.reset()
            print(f"Nouvel épisode {episode_count + 1} commencé")

except KeyboardInterrupt:
    print("\nTest interrompu par l'utilisateur")

finally:
    env.close()
    print("Environnement fermé")