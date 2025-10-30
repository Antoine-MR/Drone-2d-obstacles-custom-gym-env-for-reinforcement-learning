#!/usr/bin/env python3
"""
Script de lancement principal pour l'entraînement Eureka
avec options interactives et gestion robuste
"""

import sys
import os
import subprocess
from pathlib import Path

# Ajouter le répertoire de scripts au path
sys.path.append('training_scripts')

from fixed_eureka_training import EurekaTrainingManager
import numpy as np

def main():
    print("🚀 EUREKA DRONE TRAINING - LANCEMENT PRINCIPAL")
    print("=" * 60)
    
    BASE_DIR = r'C:\Users\antoi\Documents\cours\si5\drl\Drone-2d-obstacles-custom-gym-env-for-reinforcement-learning'
    
    print("\nOptions d'entraînement disponibles:")
    print("1. 🧪 Test rapide (1 itération, 5000 timesteps)")
    print("2. 🏃 Entraînement standard (5 itérations, 50000 timesteps)")
    print("3. 🏋️ Entraînement intensif (10 itérations, 100000 timesteps)")
    print("4. ⚙️ Configuration personnalisée")
    print("5. 📊 Monitoring des sessions existantes")
    print("6. 🔍 Voir le guide d'entraînement")
    
    try:
        choice = input("\nChoisissez une option (1-6): ").strip()
        
        if choice == "1":
            print("\n🧪 Lancement du test rapide...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=1,
                timesteps_per_iteration=5000
            )
            trainer.run_training_session()
            
        elif choice == "2":
            print("\n🏃 Lancement de l'entraînement standard...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=5,
                timesteps_per_iteration=50000
            )
            trainer.run_training_session()
            
        elif choice == "3":
            print("\n🏋️ Lancement de l'entraînement intensif...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=10,
                timesteps_per_iteration=100000
            )
            trainer.run_training_session()
            
        elif choice == "4":
            print("\n⚙️ Configuration personnalisée:")
            try:
                iterations = int(input("Nombre d'itérations (défaut: 5): ") or "5")
                timesteps = int(input("Timesteps par itération (défaut: 50000): ") or "50000")
                
                print(f"\n📋 Configuration:")
                print(f"   - Itérations: {iterations}")
                print(f"   - Timesteps par itération: {timesteps}")
                
                confirm = input("\nConfirmer ? (y/N): ").strip().lower()
                if confirm == 'y':
                    trainer = EurekaTrainingManager(
                        base_dir=BASE_DIR,
                        iterations=iterations,
                        timesteps_per_iteration=timesteps
                    )
                    trainer.run_training_session()
                else:
                    print("❌ Annulé")
                    
            except ValueError:
                print("❌ Valeurs invalides")
                
        elif choice == "5":
            print("\n📊 Lancement du monitoring...")
            subprocess.run([sys.executable, "training_monitor.py"])
            
        elif choice == "6":
            print("\n🔍 Guide d'entraînement:")
            guide_file = Path(BASE_DIR) / "TRAINING_GUIDE.md"
            if guide_file.exists():
                subprocess.run(["notepad", str(guide_file)], shell=True)
            else:
                print("❌ Guide non trouvé")
                
        else:
            print("❌ Option invalide")
            
    except KeyboardInterrupt:
        print("\n👋 Programme interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def check_prerequisites():
    """Vérifie les prérequis avant de lancer l'entraînement"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier Ollama
    ollama_paths = [
        "ollama",
        r"C:\Users\antoi\AppData\Local\Programs\Ollama\ollama.exe"
    ]
    
    ollama_found = False
    for ollama_path in ollama_paths:
        try:
            result = subprocess.run([ollama_path, "list"], capture_output=True, text=True)
            if result.returncode == 0:
                if "llama3.1:8b" in result.stdout:
                    print("✅ Ollama et llama3.1:8b disponibles")
                    ollama_found = True
                    break
                else:
                    print("⚠️ llama3.1:8b non trouvé, tentative de pull...")
                    subprocess.run([ollama_path, "pull", "llama3.1:8b"])
                    ollama_found = True
                    break
        except FileNotFoundError:
            continue
    
    if not ollama_found:
        print("❌ Ollama non installé ou non disponible")
        return False
    
    # Vérifier les packages Python
    required_packages = ["stable_baselines3", "gymnasium", "hydra"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} manquant")
    
    if missing_packages:
        print(f"\n📦 Installation des packages manquants: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
    
    print("✅ Vérification terminée")
    return True

if __name__ == "__main__":
    if check_prerequisites():
        main()
    else:
        print("❌ Prérequis non satisfaits")
        sys.exit(1)