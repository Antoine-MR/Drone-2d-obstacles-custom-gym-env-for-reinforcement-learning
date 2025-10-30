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

# Pour l'entraînement, on désactive le rendu graphique (plus rapide et évite les erreurs)
# fixed_map=True : Utilise une map fixe avec obstacles fixes et positions fixes
# initial_throw=False : Pas de lancer initial pour plus de stabilité
env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=False, render_shade=False,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=False, initial_throw=False,
            use_obstacles=True, num_obstacles=3, fixed_map=True)

# Créer le callback d'interruption
interrupt_callback = KeyboardInterruptCallback(save_path="checkpoint_agent", verbose=1)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")

print("=" * 70)
print("🎯 ENTRAÎNEMENT SUR MAP FIXE")
print("=" * 70)
print("MAP: 1 obstacle central + drone à gauche + cible à droite")
print("Position de départ: (150, 400) - Gauche")
print("Cible: (650, 400) - Droite")
print("Obstacle: Un carré 100x100 au centre (400, 400)")
print("L'agent doit contourner l'obstacle (haut ou bas) pour atteindre la cible")
print("PAS de lancer initial = drone stable au départ")
print("=" * 70)
print("Début de l'entraînement...")
print("Appuyez sur Ctrl+C pour arrêter l'entraînement et sauvegarder le modèle")
print("Checkpoints automatiques toutes les 50000 étapes")
print("TensorBoard logs dans ./tensorboard_logs/")

try:
    model.learn(total_timesteps=2000000, callback=interrupt_callback, progress_bar=True)
    # Si l'entraînement se termine normalement
    model.save('final_agent')
    print("Entraînement terminé! Modèle final sauvegardé: final_agent")
    
except KeyboardInterrupt:
    # Cette exception ne devrait normalement pas être attrapée ici
    # car elle est gérée par le signal handler
    print("bizarre chef")
    exit(1)
    pass

print("Programme terminé.")
print("Pour visualiser les métriques, lancez: tensorboard --logdir=./tensorboard_logs/")
print("Puis ouvrez: http://localhost:6006")

