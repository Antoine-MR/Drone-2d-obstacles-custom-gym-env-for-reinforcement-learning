from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import sys
import os
import signal
import time

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package'))

import drone_2d_custom_gym_env

class CurriculumCallback(BaseCallback):
    """Callback pour curriculum learning avec obstacles progressifs"""
    
    def __init__(self, save_path="curriculum_agent", verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.save_path = save_path
        self.interrupted = False
        
    def _on_step(self) -> bool:
        # Curriculum learning : augmenter difficulté progressivement
        if self.n_calls == 600000:  # 600k steps : passer à 2 obstacles
            print(f"\n🎓 PHASE 2: Passage à 2 obstacles (étape {self.n_calls})")
            self.training_env.env_method('update_obstacle_count', 2)
            checkpoint_path = f"{self.save_path}_phase1_600k"
            self.model.save(checkpoint_path)
            print(f"Checkpoint Phase 1 sauvegardé: {checkpoint_path}")
            
        elif self.n_calls == 1200000:  # 1.2M steps : passer à 3 obstacles
            print(f"\n🎓 PHASE 3: Passage à 3 obstacles (étape {self.n_calls})")
            self.training_env.env_method('update_obstacle_count', 3)
            checkpoint_path = f"{self.save_path}_phase2_1200k"
            self.model.save(checkpoint_path)
            print(f"Checkpoint Phase 2 sauvegardé: {checkpoint_path}")
            
        # Checkpoints réguliers
        if self.n_calls % 100000 == 0:
            checkpoint_path = f"{self.save_path}_step_{self.n_calls}"
            self.model.save(checkpoint_path)
            print(f"Checkpoint sauvegardé: {checkpoint_path}")
        
        return not self.interrupted
    
    def on_keyboard_interrupt(self):
        """Méthode appelée lors d'un Ctrl+C"""
        self.interrupted = True
        final_path = f"{self.save_path}_interrupted_{int(time.time())}"
        self.model.save(final_path)
        print(f"\nEntraînement interrompu! Modèle sauvegardé: {final_path}")

# Variable globale pour le callback
curriculum_callback = None

def signal_handler(sig, frame):
    """Gestionnaire de signal pour Ctrl+C"""
    if curriculum_callback:
        curriculum_callback.on_keyboard_interrupt()
    print("\nArrêt en cours... Veuillez patienter pour la sauvegarde.")

# Configuration du gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)

print("🎓 ENTRAÎNEMENT CURRICULUM LEARNING AVEC OBSTACLES")
print("=" * 60)
print("Phase 1 (0-600k steps): 1 obstacle")
print("Phase 2 (600k-1.2M steps): 2 obstacles") 
print("Phase 3 (1.2M-1.8M steps): 3 obstacles")
print("=" * 60)

# Commencer avec 1 obstacle seulement
env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=False, render_shade=False,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True,
            use_obstacles=True, num_obstacles=1)  # Commencer avec 1 obstacle

# Créer le callback curriculum
curriculum_callback = CurriculumCallback(save_path="curriculum_agent", verbose=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs_curriculum/")

print("Début de l'entraînement curriculum learning...")
print("Appuyez sur Ctrl+C pour arrêter l'entraînement et sauvegarder le modèle")

try:
    model.learn(total_timesteps=1800000, callback=curriculum_callback, progress_bar=True)
    # Si l'entraînement se termine normalement
    model.save('final_agent_curriculum')
    print("Entraînement curriculum terminé! Modèle final sauvegardé: final_agent_curriculum")
    
except KeyboardInterrupt:
    pass

print("Programme terminé.")
print("Pour visualiser les métriques: tensorboard --logdir=./tensorboard_logs_curriculum/")