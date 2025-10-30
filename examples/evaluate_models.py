"""
Script pour évaluer les différents modèles disponibles
"""
from stable_baselines3 import PPO
import gymnasium as gym
import sys
import os

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

def evaluate_model(model_path, num_episodes=5):
    """Évalue un modèle sur plusieurs épisodes"""
    
    # Créer l'environnement avec rendu pour visualiser
    env = gym.make('drone-2d-custom-v0', render_sim=True, render_path=True, render_shade=True,
                shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True,
                use_obstacles=True, num_obstacles=3)
    
    try:
        # Charger le modèle
        print(f"🔄 Chargement du modèle: {model_path}")
        model = PPO.load(model_path)
        print("✅ Modèle chargé avec succès!")
        
        # Évaluer sur plusieurs épisodes
        total_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            print(f"\n🎮 Épisode {episode + 1}/{num_episodes}")
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"   • Récompense: {episode_reward:.2f}")
            print(f"   • Étapes: {steps}")
            
        # Statistiques finales
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"\n📊 Résultats:")
        print(f"   • Récompense moyenne: {avg_reward:.2f}")
        print(f"   • Récompense min: {min(total_rewards):.2f}")
        print(f"   • Récompense max: {max(total_rewards):.2f}")
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement/évaluation: {e}")
    
    finally:
        env.close()

def main():
    print("🤖 Évaluation des modèles disponibles")
    print("=" * 40)
    
    # Liste des modèles à tester
    models_to_test = [
        ("final_agent", "Agent final (si disponible)"),
        ("checkpoint_agent_step_1800000", "Checkpoint final 1.8M étapes"),
        ("checkpoint_agent_step_1750000", "Checkpoint 1.75M étapes"),
        ("checkpoint_agent_step_1700000", "Checkpoint 1.7M étapes"),
        ("checkpoint_agent_step_1650000", "Checkpoint 1.65M étapes"),
        ("checkpoint_agent_step_1600000", "Checkpoint 1.6M étapes"),
    ]
    
    for model_path, description in models_to_test:
        if os.path.exists(model_path + ".zip"):
            print(f"\n🎯 Test du modèle: {description}")
            print(f"   Fichier: {model_path}")
            evaluate_model(model_path, num_episodes=3)
            
            input("\nAppuyez sur Entrée pour continuer avec le modèle suivant...")
        else:
            print(f"\n⚠️  Modèle non trouvé: {model_path}")

if __name__ == "__main__":
    main()