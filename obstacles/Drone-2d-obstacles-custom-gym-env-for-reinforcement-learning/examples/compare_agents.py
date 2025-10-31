from stable_baselines3 import PPO
import gymnasium as gym
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

# Configuration
NUM_TRIALS = 30
RENDER = False

# Agents à comparer
agents = {
    "Agent entraîné 1": "../agents/final_agentRandomPosition.zip",
    "Agent entraîné 2": "../agents/final_agentWithoutObstacle.zip"
}

print("=" * 80)
print("COMPARAISON D'AGENTS - 30 ESSAIS PAR AGENT")
print("=" * 80)
print(f"Test: Map fixe avec obstacle au centre, positions de départ aléatoires")
print(f"Nombre d'essais par agent: {NUM_TRIALS}")
print("=" * 80)

results = {}

for agent_name, model_path in agents.items():
    print(f"\n Test de {agent_name}...")
    
    # Charger le modèle
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"    Modèle non trouvé: {model_path}")
        results[agent_name] = None
        continue
    
    # Créer l'environnement (sans rendu pour aller vite)
    env = gym.make('drone-2d-custom-v0', render_sim=RENDER, render_path=False, render_shade=False,
                shade_distance=70, n_steps=500, n_fall_steps=10, change_target=False, initial_throw=False,
                use_obstacles=True, num_obstacles=3, fixed_map=True, random_start=True)
    
    model.set_env(env)
    
    # Statistiques
    successes = 0
    collisions_obstacle = 0
    collisions_boundary = 0
    timeouts = 0
    total_rewards = []
    total_steps = []
    
    # Lancer les essais
    for trial in range(NUM_TRIALS):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(500):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # Vérifier succès
            x, y = env.unwrapped.drone.frame_shape.body.position
            distance = np.sqrt((x - env.unwrapped.x_target)**2 + (y - env.unwrapped.y_target)**2)
            
            if distance < 30:
                successes += 1
                break
            
            if done:
                # Vérifier type d'échec
                if hasattr(env.unwrapped, 'obstacle_manager') and env.unwrapped.obstacle_manager.check_collision_with_drone(env.unwrapped.drone):
                    collisions_obstacle += 1
                elif abs(obs[6]) == 1 or abs(obs[7]) == 1:
                    collisions_boundary += 1
                break
        
        if episode_steps >= 500 and distance >= 30:
            timeouts += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
    
    env.close()
    
    # Calculer les statistiques
    results[agent_name] = {
        'successes': successes,
        'collisions_obstacle': collisions_obstacle,
        'collisions_boundary': collisions_boundary,
        'timeouts': timeouts,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_steps': np.mean(total_steps),
        'std_steps': np.std(total_steps),
        'success_rate': successes / NUM_TRIALS * 100
    }

# Afficher les résultats
print("\n" + "=" * 80)
print("RÉSULTATS FINAUX")
print("=" * 80)

for agent_name, stats in results.items():
    if stats is None:
        print(f"\n{agent_name}: MODÈLE NON TROUVÉ")
        continue
    
    print(f"\n🤖 {agent_name}")
    print(f"   {'─' * 70}")
    print(f"   Succès:              {stats['successes']}/{NUM_TRIALS} ({stats['success_rate']:.1f}%)")
    print(f"   Collisions obstacle: {stats['collisions_obstacle']}/{NUM_TRIALS} ({stats['collisions_obstacle']/NUM_TRIALS*100:.1f}%)")
    print(f"   Sorties limites:     {stats['collisions_boundary']}/{NUM_TRIALS} ({stats['collisions_boundary']/NUM_TRIALS*100:.1f}%)")
    print(f"   Timeouts:            {stats['timeouts']}/{NUM_TRIALS} ({stats['timeouts']/NUM_TRIALS*100:.1f}%)")
    print(f"   Reward moyen:        {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"   Steps moyen:         {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")

# Déterminer le gagnant
print("\n" + "=" * 80)
print("Résultat de la comparaison")
print("=" * 80)

valid_results = {name: stats for name, stats in results.items() if stats is not None}

if len(valid_results) < 2:
    print("Impossible de comparer: au moins un modèle est manquant")
else:
    agent_names = list(valid_results.keys())
    agent1_name = agent_names[0]
    agent2_name = agent_names[1]
    
    agent1_stats = valid_results[agent1_name]
    agent2_stats = valid_results[agent2_name]
    
    # Score basé sur: succès (prioritaire), puis reward, puis steps
    def calculate_score(stats):
        success_weight = 1000  # Succès est le plus important
        reward_weight = 10
        steps_penalty = 0.1
        
        score = (stats['success_rate'] * success_weight +
                stats['avg_reward'] * reward_weight -
                stats['avg_steps'] * steps_penalty)
        return score
    
    score1 = calculate_score(agent1_stats)
    score2 = calculate_score(agent2_stats)
    
    print(f"\nScore {agent1_name}: {score1:.2f}")
    print(f"Score {agent2_name}: {score2:.2f}")
    
    if score1 > score2:
        winner = agent1_name
        margin = ((score1 - score2) / score2) * 100
    elif score2 > score1:
        winner = agent2_name
        margin = ((score2 - score1) / score1) * 100
    else:
        winner = None
        margin = 0
    
    if winner:
        print(f"Le meilleur modèle est : {winner}")
        print(f"   Marge: {margin:.1f}% meilleur")
    else:
        print(f"\nÉGALITÉ: Les deux agents sont équivalents")
    
    # Analyse détaillée
    print(f"\nAnalyse:")
    if agent1_stats['success_rate'] > agent2_stats['success_rate']:
        print(f"   • {agent1_name} réussit plus souvent (+{agent1_stats['success_rate'] - agent2_stats['success_rate']:.1f}%)")
    elif agent2_stats['success_rate'] > agent1_stats['success_rate']:
        print(f"   • {agent2_name} réussit plus souvent (+{agent2_stats['success_rate'] - agent1_stats['success_rate']:.1f}%)")
    
    if agent1_stats['collisions_obstacle'] < agent2_stats['collisions_obstacle']:
        print(f"   • {agent1_name} évite mieux les obstacles (-{agent2_stats['collisions_obstacle'] - agent1_stats['collisions_obstacle']} collisions)")
    elif agent2_stats['collisions_obstacle'] < agent1_stats['collisions_obstacle']:
        print(f"   • {agent2_name} évite mieux les obstacles (-{agent1_stats['collisions_obstacle'] - agent2_stats['collisions_obstacle']} collisions)")
    
    if agent1_stats['avg_steps'] < agent2_stats['avg_steps']:
        print(f"   • {agent1_name} est plus rapide (-{agent2_stats['avg_steps'] - agent1_stats['avg_steps']:.0f} steps en moyenne)")
    elif agent2_stats['avg_steps'] < agent1_stats['avg_steps']:
        print(f"   • {agent2_name} est plus rapide (-{agent1_stats['avg_steps'] - agent2_stats['avg_steps']:.0f} steps en moyenne)")

print("\n" + "=" * 80)
