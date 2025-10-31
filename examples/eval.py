from stable_baselines3 import PPO
import gymnasium as gym
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

continuous_mode = True #if True, after completing one episode the next one will start automatically
random_action = False #if True, the agent will take actions randomly
use_fixed_map = True #if True, utilise la map fixe pour l'évaluation

render_sim = True #if True, a graphic is generated

env = gym.make('drone-2d-custom-v0', render_sim=render_sim, render_path=True, render_shade=True,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=False, initial_throw=False,
            use_obstacles=True, num_obstacles=3, fixed_map=use_fixed_map, random_start=True)

"""
Chargement du modèle entraîné
Essaie plusieurs modèles dans l'ordre de préférence
"""
# Liste des modèles à essayer dans l'ordre de préférence
model_paths = [
    "final_agentWithObstacle.zip",                              # Modèle final (s'il existe)
]

model = None
for model_path in model_paths:
    try:
        model = PPO.load(model_path)
        print(f"✅ Modèle chargé: {model_path}")
        break
    except FileNotFoundError:
        print(f"⚠️  Modèle non trouvé: {model_path}")
        continue
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {model_path}: {e}")
        continue

if model is None:
    print("❌ Aucun modèle trouvé! Veuillez d'abord entraîner un modèle.")
    exit(1)

model.set_env(env)

random_seed = int(time.time())
model.set_random_seed(random_seed)

# Nouvelle API Gymnasium : env.reset() retourne (obs, info)
obs, info = env.reset()

# Variables pour les statistiques
episode_count = 0
total_reward = 0
step_count = 0

try:
    while True:
        if render_sim:
            env.render()

        if random_action:
            action = env.action_space.sample()
        else:
            action, _states = model.predict(obs, deterministic=True)

        # Nouvelle API Gymnasium : env.step() retourne (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        step_count += 1

        if done is True:
            episode_count += 1
            print(f"📊 Épisode {episode_count} terminé:")
            print(f"   • Récompense totale: {total_reward:.2f}")
            print(f"   • Nombre d'étapes: {step_count}")
            
            if continuous_mode is True:
                obs, info = env.reset()
                total_reward = 0
                step_count = 0
            else:
                break

except KeyboardInterrupt:
    print(f"\n⚠️  Évaluation interrompue par l'utilisateur")
    print(f"📊 Statistiques finales:")
    print(f"   • Épisodes complétés: {episode_count}")
    if episode_count > 0:
        print(f"   • Récompense de l'épisode en cours: {total_reward:.2f}")
        print(f"   • Étapes de l'épisode en cours: {step_count}")

finally:
    env.close()
