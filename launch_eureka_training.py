#!/usr/bin/env python3
"""
EUREKA DRONE TRAINING - Script de lancement unifié
=====================================================

CONFIGURATION RAPIDE:
--------------------
Modifiez les variables ci-dessous pour personnaliser l'entraînement:

- ITERATIONS: Nombre d'itérations Eureka (le LLM améliore la reward à chaque fois)
  • 1 = Test rapide (~5-10 min)
  • 5 = Entraînement standard (~25-50 min)
  • 10 = Entraînement avancé (~50-100 min)

- TIMESTEPS_PER_ITERATION: Nombre de steps d'entraînement PPO par itération
  • 5000 = Test rapide (apprentissage limité)
  • 50000 = Standard (bon compromis)
  • 100000+ = Entraînement long (meilleure convergence)

- N_ENVS: Nombre d'environnements parallèles (augmente la vitesse)
  • 4 = Standard
  • 8 = Plus rapide si vous avez un bon CPU
  • 16 = Maximum recommandé

- LEARNING_RATE: Taux d'apprentissage PPO
  • 3e-4 = Standard (valeur par défaut)
  • 1e-4 = Plus lent mais plus stable
  • 1e-3 = Plus rapide mais risque d'instabilité

- N_EVAL_EPISODES: Nombre d'épisodes pour l'évaluation finale
  • 10 = Standard
  • 20+ = Plus précis mais plus long
"""

import os
import sys
import json
import time
import signal
import subprocess
import shutil
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from gymnasium import spaces

# Ajouter Ollama au PATH si nécessaire (Windows)
if os.name == 'nt':
    ollama_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama')
    if ollama_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + ollama_path

# ========== CONFIGURATION PERSONNALISABLE ==========
ITERATIONS = 1                  # Nombre d'itérations Eureka
TIMESTEPS_PER_ITERATION = 5000  # Timesteps par itération
N_ENVS = 4                      # Environnements parallèles
LEARNING_RATE = 3e-4            # Taux d'apprentissage PPO
BATCH_SIZE = 64                 # Taille des batchs
N_EPOCHS = 10                   # Époques par update PPO
CHECKPOINT_FREQ = 10000         # Fréquence de sauvegarde
N_EVAL_EPISODES = 10            # Épisodes d'évaluation
# ===================================================

def main():
    print("=" * 60)
    print("EUREKA DRONE TRAINING - LANCEMENT DIRECT")
    print("=" * 60)
    
    BASE_DIR = Path(__file__).parent.absolute()
    
    print(f"\n Configuration :")
    print(f"   Itérations Eureka : {ITERATIONS}")
    print(f"   Timesteps/itération : {TIMESTEPS}")
    print(f"   Dossier de travail : {BASE_DIR}")
    print(f"\n  Temps estimé : ~{ITERATIONS * 5}-{ITERATIONS * 10} minutes")
    print("\n Démarrage de l'entraînement...\n")
    print("=" * 60 + "\n")
    
    try:
        trainer = EurekaTrainingManager(
            base_dir=BASE_DIR,
            iterations=ITERATIONS,
            timesteps_per_iteration=TIMESTEPS
        )
        trainer.run_training_session()
        
    except KeyboardInterrupt:
        print("\n\n[!] Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
