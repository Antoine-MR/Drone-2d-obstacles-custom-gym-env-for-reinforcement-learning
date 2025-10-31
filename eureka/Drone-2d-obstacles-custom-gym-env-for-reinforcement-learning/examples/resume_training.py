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
            print(f"🔄 Checkpoint sauvegardé: {checkpoint_path}")
        
        return not self.interrupted
    
    def on_keyboard_interrupt(self):
        """Méthode appelée lors d'un Ctrl+C"""
        self.interrupted = True
        final_path = f"{self.save_path}_interrupted_{int(time.time())}"
        self.model.save(final_path)
        print(f"\n⚠️  Entraînement interrompu! Modèle sauvegardé: {final_path}")

# Variable globale pour le callback
interrupt_callback = None

def signal_handler(sig, frame):
    """Gestionnaire de signal pour Ctrl+C"""
    if interrupt_callback:
        interrupt_callback.on_keyboard_interrupt()
    print("\n⚠️  Arrêt en cours... Veuillez patienter pour la sauvegarde.")

# Configuration du gestionnaire de signal
signal.signal(signal.SIGINT, signal_handler)

# Créer l'environnement
env = gym.make('drone-2d-custom-v0', render_sim=False, render_path=False, render_shade=False,
            shade_distance=70, n_steps=500, n_fall_steps=10, change_target=True, initial_throw=True)

# Créer le callback d'interruption
interrupt_callback = KeyboardInterruptCallback(save_path="checkpoint_agent", verbose=1)

# REPRENDRE DEPUIS LE CHECKPOINT D'INTERRUPTION
checkpoint_path = "../checkpoint_agent_interrupted_1759758715"
print(f"🔄 Chargement du checkpoint: {checkpoint_path}")

try:
    model = PPO.load(checkpoint_path, env=env)
    print("✅ Checkpoint chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur lors du chargement du checkpoint: {e}")
    print("🆕 Création d'un nouveau modèle...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="../tensorboard_logs/")

# Calculer les étapes restantes
total_timesteps = 1800000
steps_completed = 456705  # Étapes déjà complétées selon votre log
remaining_steps = total_timesteps - steps_completed

print("📊 Informations de reprise:")
print(f"   • Étapes complétées: {steps_completed:,}")
print(f"   • Étapes restantes: {remaining_steps:,}")
print(f"   • Progression: {(steps_completed/total_timesteps)*100:.1f}%")
print()
print("🚀 Reprise de l'entraînement...")
print("   • Appuyez sur Ctrl+C pour arrêter et sauvegarder")
print("   • Checkpoints automatiques toutes les 50000 étapes")
print("   • TensorBoard logs dans ../tensorboard_logs/")
print()

try:
    model.learn(total_timesteps=remaining_steps, callback=interrupt_callback, progress_bar=True, reset_num_timesteps=False)
    # Si l'entraînement se termine normalement
    model.save('../final_agent_complete')
    print("✅ Entraînement terminé! Modèle final sauvegardé: ../final_agent_complete")
    
except KeyboardInterrupt:
    print("⚠️  Interruption détectée par KeyboardInterrupt")
    pass

print("🏁 Programme terminé.")
print("📊 Pour visualiser les métriques, lancez: tensorboard --logdir=../tensorboard_logs/")
print("🌐 Puis ouvrez: http://localhost:6006")