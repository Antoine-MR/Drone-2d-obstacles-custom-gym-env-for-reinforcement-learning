import gymnasium as gym
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

def compare_agents_with_obstacles(agent_with_obstacles_path, agent_without_obstacles_path, num_episodes=30):
    """
    Compare deux agents (avec et sans obstacles) testés dans un environnement AVEC obstacles sur la MAP FIXE
    """
    print(f"🔬 COMPARAISON AGENTS - MAP FIXE AVEC OBSTACLE")
    print("=" * 80)
    print(f"🤖 Agent A: {agent_with_obstacles_path}")
    print(f"🤖 Agent B: {agent_without_obstacles_path}")
    print(f"🗺️  Map: Drone (150,400) → Cible (650,400) avec obstacle central")
    print("=" * 80)
    
    # Environnement de test AVEC LA MAP FIXE
    env = gym.make('drone-2d-custom-v0', 
                   render_sim=False, render_path=False, render_shade=False,
                   n_steps=500, n_fall_steps=10, change_target=False, initial_throw=False,
                   use_obstacles=True, num_obstacles=3, fixed_map=True)
    
    def test_agent_performance(model_path, agent_name):
        """Teste un agent et retourne ses statistiques"""
        try:
            model = PPO.load(model_path)
            print(f"\n✅ {agent_name} chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur chargement {agent_name}: {e}")
            return None
        
        rewards = []
        steps = []
        collision_obstacles = 0
        out_of_bounds = 0
        timeouts = 0
        distances_to_target = []
        
        print(f"🧪 Test de {agent_name} sur {num_episodes} épisodes...")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            episode_distances = []
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Distance à la cible
                distance_x = obs[4]
                distance_y = obs[5]
                distance_to_target = np.sqrt(distance_x**2 + distance_y**2)
                episode_distances.append(distance_to_target)
                
                if done or truncated:
                    break
            
            rewards.append(episode_reward)
            steps.append(step_count)
            distances_to_target.extend(episode_distances)
            
            # Analyser cause de fin d'épisode
            if step_count < 500:
                if abs(obs[6]) == 1 or abs(obs[7]) == 1:
                    out_of_bounds += 1
                    # print(f"   🚫 Épisode {episode+1}: Sortie limites à l'étape {step_count}")
                else:
                    collision_obstacles += 1
                    # print(f"   💥 Épisode {episode+1}: Collision obstacle à l'étape {step_count}")
            else:
                timeouts += 1
                # print(f"   ✅ Épisode {episode+1}: Survie complète (500 étapes)")
        
        # Calculer statistiques
        stats = {
            'agent_name': agent_name,
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps),
            'avg_distance': np.mean(distances_to_target),
            'survival_rate': (timeouts / num_episodes) * 100,
            'collision_rate': (collision_obstacles / num_episodes) * 100,
            'bounds_rate': (out_of_bounds / num_episodes) * 100,
            'collision_count': collision_obstacles,
            'survival_count': timeouts,
            'bounds_count': out_of_bounds,
            'rewards': rewards,
            'steps': steps
        }
        
        # Affichage des statistiques de base pour cet agent
        print(f"\n📋 RÉSUMÉ {agent_name}:")
        print(f"   🎯 Épisodes survécus (500 étapes): {timeouts}/{num_episodes}")
        print(f"   💥 Collisions avec obstacles: {collision_obstacles}/{num_episodes}")
        print(f"   🚫 Sorties des limites: {out_of_bounds}/{num_episodes}")
        
        return stats
    
    # Tester les deux agents
    stats_with = test_agent_performance(agent_with_obstacles_path, "Agent AVEC obstacles")
    stats_without = test_agent_performance(agent_without_obstacles_path, "Agent SANS obstacles")
    
    if stats_with is None or stats_without is None:
        print("\n❌ ERREUR: Impossible de charger un ou plusieurs modèles")
        print("\n📋 MODÈLES RECHERCHÉS:")
        print(f"   • {agent_with_obstacles_path}.zip")
        print(f"   • {agent_without_obstacles_path}.zip")
        print("\n💡 SOLUTIONS:")
        print("   • Vérifiez que l'entraînement est terminé")
        print("   • Vérifiez les noms de fichiers")
        print("   • Les modèles doivent être dans le dossier examples/")
        env.close()
        return None, None
    
    # Affichage comparatif
    print(f"\n📊 RÉSULTATS COMPARATIFS:")
    print("=" * 80)
    print(f"{'Métrique':<25} {'Agent AVEC':<15} {'Agent SANS':<15} {'Différence':<15}")
    print("-" * 80)
    
    metrics = [
        ('Récompense moyenne', 'avg_reward', '.2f'),
        ('Étapes moyennes', 'avg_steps', '.1f'),
        ('Distance cible', 'avg_distance', '.3f'),
        ('Taux survie (%)', 'survival_rate', '.1f'),
        ('Collision obstacles (%)', 'collision_rate', '.1f'),
        ('Sortie limites (%)', 'bounds_rate', '.1f')
    ]
    
    for metric_name, key, fmt in metrics:
        val_with = stats_with[key]
        val_without = stats_without[key]
        diff = val_with - val_without
        
        # Formatage corrigé - enlever les deux points du début
        format_spec = fmt[1:] if fmt.startswith(':') else fmt
        val_with_str = f"{val_with:{format_spec}}"
        val_without_str = f"{val_without:{format_spec}}"
        diff_str = f"{diff:+{format_spec}}"
        
        print(f"{metric_name:<25} {val_with_str:<15} {val_without_str:<15} {diff_str:<15}")
    
    # Section spéciale pour les RÉUSSITES
    print(f"\n🎯 FOCUS: TAUX DE RÉUSSITE (Survie 500 étapes)")
    print("-" * 50)
    success_with = stats_with['survival_count']
    success_without = stats_without['survival_count']
    success_diff = success_with - success_without
    
    print(f"Agent final_agent:              {success_with}/{num_episodes} réussites ({stats_with['survival_rate']:.1f}%)")
    print(f"Agent final_agentWithoutObstacle: {success_without}/{num_episodes} réussites ({stats_without['survival_rate']:.1f}%)")
    print(f"Différence:                     {success_diff:+d} réussites")
    
    if success_diff > 0:
        print(f"✅ L'agent final_agent réussit {success_diff} fois de plus sur la map!")
        print(f"   → Entraîné avec obstacles = meilleure navigation")
    elif success_diff == 0:
        print(f"⚖️  Les deux agents ont le même taux de réussite")
    else:
        print(f"⚠️  L'agent final_agentWithoutObstacle réussit {abs(success_diff)} fois de plus")
    
    # Section collisions obstacles
    print(f"\n💥 FOCUS: COLLISIONS AVEC L'OBSTACLE")
    print("-" * 50)
    collision_with = stats_with['collision_count']
    collision_without = stats_without['collision_count']
    collision_reduction = collision_without - collision_with
    
    print(f"Agent final_agent:              {collision_with}/{num_episodes} collisions")
    print(f"Agent final_agentWithoutObstacle: {collision_without}/{num_episodes} collisions")
    
    if collision_reduction > 0:
        reduction_pct = (collision_reduction/collision_without*100) if collision_without > 0 else 0
        print(f"Réduction:                      {collision_reduction:+d} collisions évitées (-{reduction_pct:.1f}%)")
        print(f"✅ L'agent entraîné avec obstacles évite mieux l'obstacle!")
    elif collision_reduction == 0:
        print(f"⚖️  Même nombre de collisions")
    else:
        print(f"⚠️  {abs(collision_reduction)} collisions de plus pour l'agent entraîné")
    
    # Analyse des résultats
    print(f"\n🔍 ANALYSE COMPARATIVE:")
    print("=" * 60)
    
    survival_improvement = stats_with['survival_rate'] - stats_without['survival_rate']
    collision_improvement = stats_without['collision_rate'] - stats_with['collision_rate']
    
    # Analyse de la survie
    print(f"📊 Taux de survie:")
    if survival_improvement > 20:
        print(f"   � EXCELLENT: +{survival_improvement:.1f}% de réussite")
        print(f"   → L'agent entraîné avec obstacles est BEAUCOUP plus performant!")
    elif survival_improvement > 10:
        print(f"   ✅ TRÈS BIEN: +{survival_improvement:.1f}% de réussite")
        print(f"   → L'entraînement avec obstacles a porté ses fruits")
    elif survival_improvement > 0:
        print(f"   👍 BIEN: +{survival_improvement:.1f}% de réussite")
        print(f"   → Légère amélioration grâce à l'entraînement")
    elif survival_improvement == 0:
        print(f"   ⚖️  ÉGAL: Même performance")
    else:
        print(f"   ⚠️  SURPRENANT: {survival_improvement:.1f}% (agent sans obstacles meilleur)")
        print(f"   → Possible que l'agent sans obstacles ait eu plus d'entraînement")
    
    # Analyse des collisions
    print(f"\n💥 Évitement des collisions:")
    if collision_improvement > 20:
        print(f"   🛡️  EXCELLENT: {collision_improvement:.1f}% moins de collisions")
        print(f"   → L'agent a bien appris à éviter les obstacles!")
    elif collision_improvement > 10:
        print(f"   👍 BIEN: {collision_improvement:.1f}% moins de collisions")
        print(f"   → Amélioration notable de la navigation")
    elif collision_improvement > 0:
        print(f"   ↗️  LÉGÈRE AMÉLIORATION: {collision_improvement:.1f}% moins de collisions")
    elif collision_improvement == 0:
        print(f"   ⚖️  ÉGAL: Même nombre de collisions")
    else:
        print(f"   ⚠️  {abs(collision_improvement):.1f}% plus de collisions")
        print(f"   → L'entraînement n'a pas amélioré l'évitement")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    if collision_improvement < 10:
        print("🔄 L'agent avec obstacles peut encore s'améliorer:")
        print("   • Plus d'entraînement (augmenter timesteps)")
        print("   • Penalty collision plus élevée (-15 au lieu de -10)")
        print("   • Récompense pour évitement d'obstacles")
    
    if stats_with['survival_rate'] < 20:
        print("� Curriculum learning recommandé:")
        print("   • Commencer avec 1 obstacle, puis 2, puis 3")
        print("   • Obstacles plus grands au début")
    
    # Verdict final - QUI GAGNE?
    print(f"\n🏆 VERDICT FINAL:")
    print("=" * 60)
    
    if success_with > success_without:
        winner = "final_agent (entraîné AVEC obstacles)"
        margin = success_with - success_without
        print(f"🥇 GAGNANT: {winner}")
        print(f"   ✅ {success_with} réussites VS {success_without} réussites")
        print(f"   → Marge de victoire: +{margin} réussites sur {num_episodes} essais")
        print(f"   � L'entraînement avec obstacles a été efficace!")
    elif success_without > success_with:
        winner = "final_agentWithoutObstacle (entraîné SANS obstacles)"
        margin = success_without - success_with
        print(f"🥇 GAGNANT: {winner}")
        print(f"   ✅ {success_without} réussites VS {success_with} réussites")
        print(f"   → Marge de victoire: +{margin} réussites sur {num_episodes} essais")
        print(f"   ⚠️  Résultat surprenant!")
    else:
        print(f"🤝 ÉGALITÉ PARFAITE")
        print(f"   → Les deux agents ont {success_with} réussites chacun")
        print(f"   → Performance identique sur {num_episodes} essais")
    
    env.close()
    return stats_with, stats_without

def list_available_models():
    """Liste les modèles disponibles dans le répertoire"""
    import glob
    
    print("📂 MODÈLES DISPONIBLES:")
    print("-" * 40)
    
    # Chercher tous les fichiers .zip
    model_files = glob.glob("*.zip")
    checkpoint_files = glob.glob("checkpoint_*.zip")
    
    if model_files:
        for model in sorted(model_files):
            print(f"   ✅ {model}")
    else:
        print("   ❌ Aucun modèle trouvé dans le répertoire courant")
        print("   💡 Assurez-vous d'être dans le dossier examples/")
        print("   💡 L'entraînement doit être terminé pour créer les fichiers .zip")
    
    return model_files

if __name__ == "__main__":
    print("🤖 COMPARAISON AGENTS: AVEC vs SANS obstacles")
    print("=" * 60)
    
    # D'abord, lister les modèles disponibles
    available_models = list_available_models()
    
    if not available_models:
        print("\n⚠️  Aucun modèle disponible pour l'analyse")
        print("\n📚 ÉTAPES NÉCESSAIRES:")
        print("   1. Terminer l'entraînement avec obstacles")
        print("   2. S'assurer d'avoir 'final_agentWithoutObstacle.zip'")
        print("   3. Relancer ce script")
        exit(1)
    
    # Chemins des modèles
    agent_with_obstacles = "final_agent"  # Agent entraîné avec obstacles
    agent_without_obstacles = "final_agentWithoutObstacle"  # Agent entraîné sans obstacles
    
    # 1. Comparaison quantitative sur 30 essais
    print("\n🔬 ANALYSE COMPARATIVE - 30 ESSAIS CHACUN:")
    result = compare_agents_with_obstacles(
        agent_with_obstacles, 
        agent_without_obstacles, 
        num_episodes=30
    )
    
    if result is None:
        print("\n⚠️  Analyse interrompue - modèles non disponibles")
        exit(1)
    
    stats_with, stats_without = result
    
    print("\n" + "=" * 70)
    print("🎉 ANALYSE TERMINÉE!")
    print("=" * 70)
    print(f"✅ 30 essais par agent sur la map fixe avec obstacle")
    print(f"🗺️  Configuration: Drone (150,400) → Cible (650,400) + obstacle central")
    print("=" * 70)