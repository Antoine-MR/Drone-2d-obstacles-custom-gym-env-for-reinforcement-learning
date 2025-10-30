#!/usr/bin/env python3
"""
Script de lancement direct Eureka sans menu interactif
Lance automatiquement un entraînement quick test (1 iteration, 5000 timesteps)
"""

import sys
import os
from pathlib import Path

# Ajouter Ollama au PATH si nécessaire
if os.name == 'nt':  # Windows
    ollama_path = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama')
    if ollama_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + ollama_path

sys.path.append('training_scripts')

from fixed_eureka_training import EurekaTrainingManager

def main():
    print("=" * 60)
    print("EUREKA DRONE TRAINING - LANCEMENT DIRECT")
    print("=" * 60)
    
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Configuration : modifiez ces valeurs selon vos besoins
    ITERATIONS = 10          # Nombre d'itérations Eureka
    TIMESTEPS = 5000        # Timesteps par itération
    
    print(f"\nConfiguration :")
    print(f"  - Itérations : {ITERATIONS}")
    print(f"  - Timesteps/itération : {TIMESTEPS}")
    print(f"  - Dossier : {BASE_DIR}")
    print("\nDémarrage de l'entraînement...\n")
    
    try:
        trainer = EurekaTrainingManager(
            base_dir=BASE_DIR,
            iterations=ITERATIONS,
            timesteps_per_iteration=TIMESTEPS
        )
        
        trainer.run_training_session()
        
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT TERMINÉ !")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n[!] Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
