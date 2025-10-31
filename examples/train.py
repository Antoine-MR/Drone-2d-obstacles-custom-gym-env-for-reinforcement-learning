from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import sys
import os
import signal
import time

# Ajouter le chemin vers le package local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'drone_2d_custom_gym_env_package/agents'))

import drone_2d_custom_gym_env

class KeyboardInterruptCallback(BaseCallback):
    """Callback pour gérer l'arrêt manuel avec Ctrl+C"""
    
    def __init__(self, save_path="checkpoint_agent", verbose=0):
        super(KeyboardInterruptCallback, self).__init__(verbose)
        self.save_path = save_path
        self.interrupted = False
        
    def _on_step(self) -> bool:
        # Sauvegarder périodiquement (toutes les 50000 étapes)
        if self.n_calls % 50000 == 0:
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
interrupt_callback = None

def signal_handler(sig, frame):
    """Gestionnaire de signal pour Ctrl+C"""
    if interrupt_callback:
        interrupt_callback.on_keyboard_interrupt()
    print("\nArrêt en cours... Veuillez patienter pour la sauvegarde.")

# Configuration du gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)

env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=False, render_shade=False,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=False, initial_throw=False,
            use_obstacles=True, num_obstacles=3, fixed_map=True, random_start=True)

# Créer le callback d'interruption
interrupt_callback = KeyboardInterruptCallback(save_path="checkpoint_agent", verbose=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

try:
    model.learn(total_timesteps=4000000, callback=interrupt_callback, progress_bar=True)
    # Si l'entraînement se termine normalement
    model.save('final_agent')
    print("Entraînement terminé! Modèle final sauvegardé: final_agent")
    
except KeyboardInterrupt:
    # Cette exception ne devrait normalement pas être attrapée ici
    # car elle est gérée par le signal handler
    exit(1)
    pass

print("Programme terminé.")
print("Pour visualiser les métriques, lancez: tensorboard --logdir=./tensorboard_logs/")
print("Puis ouvrez: http://localhost:6006")

