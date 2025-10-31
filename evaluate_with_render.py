#!/usr/bin/env python
"""
Script pour evaluer un modele PPO entraine avec rendu visuel (render=True).
Permet de visualiser le comportement du drone dans l'environnement.

Configuration des conditions de crash (ligne 15):
    DISABLE_ANGLE_CRASH = False  # Mettre True pour ignorer les crashs dus a l'angle (seule la position compte)

Configuration de la vitesse de la simulation (ligne 18):
    RENDER_FPS = 30  # Frames par seconde pour le rendu (30 = normal, 10 = lent, 60 = rapide, 0 = aussi vite que possible)

Usage:
    python evaluate_with_render.py --model_path <chemin_vers_model.zip> --reward_file <chemin_vers_reward_iter.py> --n_episodes <nombre>

Example:
    python evaluate_with_render.py --model_path "training_sessions/eureka_session_20251031_131836/models/model_iter2_final.zip" --reward_file "training_sessions/eureka_session_20251031_131836/reward_functions/reward_iter2.py" --n_episodes 10
"""

# ===== CONFIGURATION DES CONDITIONS DE CRASH =====
DISABLE_ANGLE_CRASH = True  # Si True, l'angle ne provoque pas de crash (seule la position hors limites crashe)
# ==================================================

# ===== CONFIGURATION DE LA VITESSE DE SIMULATION =====
RENDER_FPS = 60  # Frames par seconde (30 = normal, 10 = lent pour bien voir, 60 = rapide, 0 = illimite)
# ======================================================

import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Ajouter les chemins necessaires
sys.path.insert(0, str(Path(__file__).parent / "Eureka_for_carracing-master" / "Eureka_for_carracing-master" / "rl-baselines3-zoo"))
sys.path.insert(0, str(Path(__file__).parent / "Eureka_for_carracing-master" / "Eureka_for_carracing-master" / "eureka" / "envs" / "drone_2d"))
sys.path.insert(0, str(Path(__file__).parent / "drone_2d_custom_gym_env_package"))

from stable_baselines3 import PPO
from drone_2d_obs import Drone2DCustom, compute_reward
from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
import shutil
import gymnasium as gym
import time  # Pour ajouter des delais si necessaire


class Drone2DCustomEval(Drone2DCustom):
    """Wrapper pour l'evaluation avec conditions de crash personnalisables."""
    
    def __init__(self, render_sim=False, render_path=False, render_shade=False, shade_distance=70,
                 n_steps=10000, n_fall_steps=10, change_target=False, initial_throw=False,
                 disable_angle_crash=False):
        """
        Args:
            disable_angle_crash: Si True, l'angle ne provoque pas de crash (seule la position hors limites crashe)
            Tous les autres parametres sont passes a Drone2DCustom
        """
        super().__init__(render_sim=render_sim, render_path=render_path, render_shade=render_shade,
                        shade_distance=shade_distance, n_steps=n_steps, n_fall_steps=n_fall_steps,
                        change_target=change_target, initial_throw=initial_throw)
        self.disable_angle_crash = disable_angle_crash
        self._init_disable_angle_crash = disable_angle_crash  # Sauvegarde pour le reset
    
    def step(self, action):
        # Appeler le step du parent Drone2DCustom (qui inclut la reward function Eureka)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Si disable_angle_crash est active, recalculer terminated
        if self.disable_angle_crash and terminated:
            # Verifier si c'est un crash de position (pas d'angle)
            is_position_crash = np.abs(obs['position'][0]) == 1 or np.abs(obs['position'][1]) == 1
            is_timeout = self.current_time_step >= self.max_time_steps
            
            # Si ce n'est ni un crash de position ni un timeout, continuer
            if not is_position_crash and not is_timeout:
                terminated = False
                self.done = False  # IMPORTANT: reset le flag interne
                info['crash'] = False
        
        return obs, reward, terminated, truncated, info


def load_reward_function(reward_file_path):
    """Charge la fonction de recompense depuis un fichier Python."""
    print(f"[INFO] Chargement de la fonction de recompense depuis: {reward_file_path}")
    
    # Copier le fichier reward dans drone_2d_obs.py
    drone_2d_obs_path = Path(__file__).parent / "Eureka_for_carracing-master" / "Eureka_for_carracing-master" / "eureka" / "envs" / "drone_2d" / "drone_2d_obs.py"
    
    # Lire le contenu du fichier reward
    with open(reward_file_path, 'r', encoding='utf-8') as f:
        reward_content = f.read()
    
    # Lire le fichier drone_2d_obs.py actuel
    with open(drone_2d_obs_path, 'r', encoding='utf-8') as f:
        obs_content = f.read()
    
    # Trouver et remplacer la fonction compute_reward
    import re
    pattern = r'def compute_reward\(self[^}]*?\n(?:\s{4}.*\n)*?\s{4}return.*'
    
    if 'def compute_reward(' in reward_content:
        # Extraire juste la fonction compute_reward du fichier reward
        reward_match = re.search(pattern, reward_content, re.MULTILINE)
        if reward_match:
            new_compute_reward = reward_match.group(0)
            # Remplacer dans obs_content
            obs_content_updated = re.sub(pattern, new_compute_reward, obs_content, count=1, flags=re.MULTILINE)
            
            # Ecrire le fichier mis a jour
            with open(drone_2d_obs_path, 'w', encoding='utf-8') as f:
                f.write(obs_content_updated)
            
            print("[INFO] Fonction de recompense chargee avec succes.")
        else:
            print("[WARNING] Impossible de trouver la fonction compute_reward dans le fichier reward.")
    else:
        print("[WARNING] Le fichier reward ne contient pas de fonction compute_reward.")


def evaluate_model(model_path, reward_file_path, n_episodes=10, render=True, use_training_params=True):
    """Evalue un modele avec rendu visuel.
    
    Args:
        use_training_params: Si True, utilise les memes parametres que l'entrainement (initial_throw=True)
                            Si False, utilise des parametres plus favorables (initial_throw=False)
    """
    
    print("\n" + "="*60)
    print("EVALUATION DU MODELE AVEC RENDU VISUEL")
    print("="*60)
    print(f"Modele: {model_path}")
    print(f"Reward function: {reward_file_path}")
    print(f"Nombre d'episodes: {n_episodes}")
    print(f"Rendu visuel: {render}")
    print("="*60 + "\n")
    
    # Charger la fonction de recompense
    if reward_file_path:
        load_reward_function(reward_file_path)
    
    # Creer l'environnement
    if use_training_params:
        print("[INFO] Utilisation des parametres d'entrainement (initial_throw=True, n_steps=1000)")
        print(f"[INFO] Crash par angle: {'DESACTIVE' if DISABLE_ANGLE_CRASH else 'ACTIVE'} (configure dans drone_2d_env.py)")
        env = Drone2DCustom(
            render_sim=render, 
            render_path=True, 
            render_shade=False,
            n_steps=1000,
            initial_throw=True,
            n_fall_steps=10
        )
    else:
        print("[INFO] Utilisation de parametres favorables (initial_throw=False, n_steps=2000)")
        print(f"[INFO] Crash par angle: {'DESACTIVE' if DISABLE_ANGLE_CRASH else 'ACTIVE'} (configure dans drone_2d_env.py)")
        env = Drone2DCustom(
            render_sim=render, 
            render_path=True, 
            render_shade=False,
            n_steps=2000,
            initial_throw=False,
            n_fall_steps=0
        )
    
    # Charger le modele
    print(f"[INFO] Chargement du modele depuis: {model_path}")
    model = PPO.load(model_path, env=env)
    print("[INFO] Modele charge avec succes.\n")
    
    # Statistiques
    episode_rewards = []
    episode_lengths = []
    successes = []
    crashes = []
    distances_to_target = []
    
    print("Debut de l'evaluation...\n")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_distances = []
        
        print(f"[Episode {episode + 1}/{n_episodes}] Demarrage...")
        
        step_count = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            # Log position chaque 10 steps pour debug
            if step_count % 10 == 0:
                x_px = (obs['position'][0] + 1) * 400.0
                y_px = (obs['position'][1] + 1) * 400.0
                print(f"  Step {step_count}: pos=({x_px:.1f}, {y_px:.1f})px, angle={obs['angle'][0]*90:.1f}deg, done={done}")
            
            # Limiter le framerate si render est active et FPS > 0
            if render and RENDER_FPS > 0:
                time.sleep(1.0 / RENDER_FPS)
            
            if 'distance_to_target' in info:
                episode_distances.append(info['distance_to_target'])
            
            # Debug: afficher les infos au moment du crash
            if done and episode == 0:  # Seulement pour le premier episode
                # Convertir position normalisee en pixels
                pos_x_pixels = (obs['position'][0] + 1) * 400.0
                pos_y_pixels = (obs['position'][1] + 1) * 400.0
                angle_degrees = obs['angle'][0] * 90.0
                
                print(f"  [DEBUG] Crash a l'etape {episode_length}:")
                print(f"    Position normalisee: x={obs['position'][0]:.3f}, y={obs['position'][1]:.3f}")
                print(f"    Position reelle: x={pos_x_pixels:.1f}px, y={pos_y_pixels:.1f}px (limite: 0-800px)")
                print(f"    Angle: {angle_degrees:.1f} degres (limite: -90 a +90 degres)")
                print(f"    Velocity: vx={obs['velocity'][0]:.3f}, vy={obs['velocity'][1]:.3f}")
                print(f"    Terminated: {terminated}, Truncated: {truncated}")
                
                # Identifier la raison du crash
                if abs(obs['position'][0]) >= 1.0:
                    print(f"    -> CRASH: Position X hors limites ({'gauche' if obs['position'][0] < 0 else 'droite'})")
                if abs(obs['position'][1]) >= 1.0:
                    print(f"    -> CRASH: Position Y hors limites ({'bas' if obs['position'][1] < 0 else 'haut'})")
                if abs(obs['angle'][0]) >= 1.0:
                    print(f"    -> CRASH: Angle trop incline ({angle_degrees:.1f} degres)")
        
        # Collecter les statistiques
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        success = info.get('success', False)
        crash = info.get('crash', False)
        final_distance = info.get('distance_to_target', np.nan)
        
        successes.append(1 if success else 0)
        crashes.append(1 if crash else 0)
        distances_to_target.append(final_distance)
        
        # Afficher le resultat de l'episode
        status = ""
        if success:
            status = "[SUCCES]"
        elif crash:
            status = "[CRASH]"
        else:
            status = "[TERMINE]"
        
        print(f"  {status} Reward: {episode_reward:.2f} | Longueur: {episode_length} steps | Distance finale: {final_distance:.2f} px\n")
    
    env.close()
    
    # Calculer et afficher les statistiques finales
    print("\n" + "="*60)
    print("RESULTATS DE L'EVALUATION")
    print("="*60)
    print(f"Nombre d'episodes: {n_episodes}")
    print(f"\nRecompense moyenne: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Recompense min: {np.min(episode_rewards):.2f}")
    print(f"Recompense max: {np.max(episode_rewards):.2f}")
    print(f"\nTaux de succes: {np.mean(successes) * 100:.1f}% ({np.sum(successes)}/{n_episodes})")
    print(f"Taux de crash: {np.mean(crashes) * 100:.1f}% ({np.sum(crashes)}/{n_episodes})")
    print(f"\nDistance moyenne au but: {np.nanmean(distances_to_target):.2f} px")
    print(f"Longueur moyenne des episodes: {np.mean(episode_lengths):.1f} steps")
    print("="*60 + "\n")
    
    return {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'success_rate': float(np.mean(successes)),
        'crash_rate': float(np.mean(crashes)),
        'avg_distance_to_target': float(np.nanmean(distances_to_target)),
        'avg_episode_length': float(np.mean(episode_lengths)),
        'episode_rewards': [float(r) for r in episode_rewards]
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluer un modele PPO avec rendu visuel')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Chemin vers le fichier .zip du modele')
    parser.add_argument('--reward_file', type=str, default=None,
                       help='Chemin vers le fichier reward_iterX.py (optionnel)')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Nombre d\'episodes a evaluer (defaut: 10)')
    parser.add_argument('--no_render', action='store_true',
                       help='Desactiver le rendu visuel')
    parser.add_argument('--easy_mode', action='store_true',
                       help='Utiliser des parametres favorables (initial_throw=False, plus de temps)')
    
    args = parser.parse_args()
    
    # Verifier que le modele existe
    if not os.path.exists(args.model_path):
        print(f"[ERREUR] Le fichier modele n'existe pas: {args.model_path}")
        sys.exit(1)
    
    # Verifier que le fichier reward existe si specifie
    if args.reward_file and not os.path.exists(args.reward_file):
        print(f"[ERREUR] Le fichier reward n'existe pas: {args.reward_file}")
        sys.exit(1)
    
    # Evaluer le modele
    results = evaluate_model(
        model_path=args.model_path,
        reward_file_path=args.reward_file,
        n_episodes=args.n_episodes,
        render=not args.no_render,
        use_training_params=not args.easy_mode
    )
    
    print("[INFO] Evaluation terminee.")


if __name__ == "__main__":
    main()
