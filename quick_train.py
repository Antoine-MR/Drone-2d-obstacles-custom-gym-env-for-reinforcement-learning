#!/usr/bin/env python3
"""
🚀 LANCEMENT RAPIDE - ENTRAÎNEMENT EUREKA COMPLET
Script optimisé pour lancer directement un entraînement standard
"""

import sys
import os
from pathlib import Path

# Configuration
BASE_DIR = r'C:\Users\antoi\Documents\cours\si5\drl\Drone-2d-obstacles-custom-gym-env-for-reinforcement-learning'
ITERATIONS = 5
TIMESTEPS_PER_ITERATION = 50000

def main():
    print("🚀 EUREKA TRAINING - LANCEMENT RAPIDE")
    print("=" * 50)
    print(f"📁 Répertoire: {BASE_DIR}")
    print(f"🔄 Itérations: {ITERATIONS}")
    print(f"⏱️ Timesteps par itération: {TIMESTEPS_PER_ITERATION}")
    print(f"📊 Total estimé: ~{(ITERATIONS * 5)}-{(ITERATIONS * 10)} minutes")
    print("=" * 50)
    
    # Confirmation
    response = input("\n🤔 Lancer l'entraînement complet ? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Entraînement annulé")
        return
    
    print("\n🏃 Lancement de l'entraînement...")
    
    # Import et lancement
    sys.path.append('training_scripts')
    
    try:
        from fixed_eureka_training import EurekaTrainingManager
        
        trainer = EurekaTrainingManager(
            base_dir=BASE_DIR,
            iterations=ITERATIONS,
            timesteps_per_iteration=TIMESTEPS_PER_ITERATION
        )
        
        trainer.run_training_session()
        
    except KeyboardInterrupt:
        print("\n⏹️ Entraînement interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()