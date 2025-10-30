#!/usr/bin/env python3

import sys
import os
import subprocess
from pathlib import Path

sys.path.append('training_scripts')

from fixed_eureka_training import EurekaTrainingManager
import numpy as np

def main():
    print("EUREKA DRONE TRAINING - MAIN LAUNCHER")
    print("=" * 60)
    
    BASE_DIR = r'C:\Users\antoi\Documents\cours\si5\drl\Drone-2d-obstacles-custom-gym-env-for-reinforcement-learning'
    
    print("\nAvailable training options:")
    print("1. Quick test (1 iteration, 5000 timesteps)")
    print("2. Standard training (5 iterations, 50000 timesteps)")
    print("3. Intensive training (10 iterations, 100000 timesteps)")
    print("4. Custom configuration")
    print("5. Monitor existing sessions")
    print("6. View training guide")
    
    try:
        choice = input("\nChoose an option (1-6): ").strip()
        
        if choice == "1":
            print("\nLaunching quick test...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=1,
                timesteps_per_iteration=5000
            )
            trainer.run_training_session()
            
        elif choice == "2":
            print("\nLaunching standard training...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=5,
                timesteps_per_iteration=50000
            )
            trainer.run_training_session()
            
        elif choice == "3":
            print("\nLaunching intensive training...")
            trainer = EurekaTrainingManager(
                base_dir=BASE_DIR,
                iterations=10,
                timesteps_per_iteration=100000
            )
            trainer.run_training_session()
            
        elif choice == "4":
            print("\nCustom configuration:")
            try:
                iterations = int(input("Number of iterations (default: 5): ") or "5")
                timesteps = int(input("Timesteps per iteration (default: 50000): ") or "50000")
                
                print(f"\nConfiguration:")
                print(f"   - Iterations: {iterations}")
                print(f"   - Timesteps per iteration: {timesteps}")
                
                confirm = input("\nConfirm? (y/N): ").strip().lower()
                if confirm == 'y':
                    trainer = EurekaTrainingManager(
                        base_dir=BASE_DIR,
                        iterations=iterations,
                        timesteps_per_iteration=timesteps
                    )
                    trainer.run_training_session()
                else:
                    print("Cancelled")
                    
            except ValueError:
                print("Invalid values")
                
        elif choice == "5":
            print("\nLaunching monitoring...")
            subprocess.run([sys.executable, "training_monitor.py"])
            
        elif choice == "6":
            print("\nTraining guide:")
            guide_file = Path(BASE_DIR) / "TRAINING_GUIDE.md"
            if guide_file.exists():
                subprocess.run(["notepad", str(guide_file)], shell=True)
            else:
                print("Guide not found")
                
        else:
            print("Invalid option")
            
    except KeyboardInterrupt:
        print("\n👋 Programme interrompu par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

def check_prerequisites():
    print("Checking prerequisites...")
    
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
                    print("Ollama and llama3.1:8b available")
                    ollama_found = True
                    break
                else:
                    print("llama3.1:8b not found, attempting to pull...")
                    subprocess.run([ollama_path, "pull", "llama3.1:8b"])
                    ollama_found = True
                    break
        except FileNotFoundError:
            continue
    
    if not ollama_found:
        print("Ollama not installed or not available")
        return False
    
    required_packages = ["stable_baselines3", "gymnasium", "hydra"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"{package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} missing")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
    
    print("Verification completed")
    return True

if __name__ == "__main__":
    if check_prerequisites():
        main()
    else:
        print("Prerequisites not satisfied")
        sys.exit(1)