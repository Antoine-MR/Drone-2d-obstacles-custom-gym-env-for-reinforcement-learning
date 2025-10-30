import gymnasium as gym
import sys
import os

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

print("=" * 70)
print("🗺️  TEST DE LA MAP FIXE")
print("=" * 70)
print("Cette map est utilisée pour l'entraînement.")
print("Le drone démarre toujours à la même position.")
print("Les obstacles sont toujours aux mêmes endroits.")
print("Appuyez sur Ctrl+C pour quitter.")
print("=" * 70)

# Test avec visualisation et la map fixe
env = gym.make('drone-2d-custom-v0', 
               render_sim=True,          # Activer le rendu
               render_path=True,         # Afficher la trajectoire
               render_shade=True,        # Afficher les ombres
               shade_distance=70, 
               n_steps=500, 
               n_fall_steps=10, 
               change_target=False,      # Target fixe
               initial_throw=False,      # PAS de lancer initial = plus stable
               use_obstacles=True,       # Obstacles activés
               fixed_map=True)           # 🎯 MAP FIXE

obs, info = env.reset()
print(f"\n📍 Configuration de la map:")
print(f"   Drone départ: (150, 400) - Gauche, mi-hauteur")
print(f"   Cible: (650, 400) - Droite, mi-hauteur")
print(f"   Obstacle: 1 carré central 100x100 au milieu (400, 400)")
print(f"   PAS de lancer initial = drone stable au départ")
print(f"\n▶️  L'agent fait des actions aléatoires pour visualiser...")

episode_count = 0
step_count = 0

try:
    while True:
        # Actions aléatoires pour tester (remplacez par votre agent entraîné)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        step_count += 1
        
        if done:
            if step_count < 500:
                # Episode terminé prématurément (collision ou sortie)
                if abs(obs[6]) == 1 or abs(obs[7]) == 1:
                    print(f"❌ Episode {episode_count + 1} - Étape {step_count}: Sortie des limites")
                else:
                    print(f"💥 Episode {episode_count + 1} - Étape {step_count}: Collision avec obstacle!")
            else:
                # Episode complet
                print(f"✅ Episode {episode_count + 1} - Étape {step_count}: Episode complet")
            
            episode_count += 1
            step_count = 0
            obs, info = env.reset()

except KeyboardInterrupt:
    print("\n\n" + "=" * 70)
    print("Test interrompu par l'utilisateur")
    print("=" * 70)

finally:
    env.close()
    print("Environnement fermé")
