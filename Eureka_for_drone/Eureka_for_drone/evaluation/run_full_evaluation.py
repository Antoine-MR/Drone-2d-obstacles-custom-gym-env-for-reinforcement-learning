#!/usr/bin/env python3
"""
Complete evaluation pipeline for Eureka-optimized drone models
"""

import os
import sys
from pathlib import Path
import subprocess

def main():
    """Complete evaluation pipeline"""
    print("🎯 EUREKA DRONE MODEL EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Analyze Eureka results
    print("\n1️⃣ ANALYZING EUREKA RESULTS...")
    print("-" * 40)
    
    try:
        subprocess.run([sys.executable, "analyze_results.py"], check=True)
        print("✅ Eureka analysis completed")
    except subprocess.CalledProcessError:
        print("❌ Error in Eureka analysis")
        return
    except FileNotFoundError:
        print("❌ analyze_results.py not found")
        return
    
    # Step 2: Ask user if they want to train an agent
    print("\n2️⃣ TRAINING AGENT WITH OPTIMIZED REWARD...")
    print("-" * 40)
    
    response = input("Do you want to train an agent with the optimized reward? (y/n): ")
    if response.lower().startswith('y'):
        try:
            subprocess.run([sys.executable, "train_optimized_agent.py"], check=True)
            print("✅ Agent training completed")
        except subprocess.CalledProcessError:
            print("❌ Error in agent training")
            return
        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
    else:
        print("⏭️ Skipping agent training")
    
    # Step 3: Evaluate trained model
    print("\n3️⃣ EVALUATING TRAINED MODEL...")
    print("-" * 40)
    
    models_dir = Path("./models/")
    if not models_dir.exists() or not list(models_dir.glob("*.zip")):
        print("❌ No trained models found. Please train a model first.")
        return
    
    try:
        subprocess.run([sys.executable, "evaluate_model.py", "--episodes", "10"], check=True)
        print("✅ Model evaluation completed")
    except subprocess.CalledProcessError:
        print("❌ Error in model evaluation")
        return
    
    # Step 4: Summary
    print("\n🎉 EVALUATION PIPELINE COMPLETED!")
    print("="*60)
    print("📊 Check the following files for results:")
    print("   • evaluation_results.png - Performance comparison plots")
    print("   • ./logs/ - Detailed evaluation logs")
    print("   • ./models/ - Trained model files")
    print("   • ./tensorboard_logs/ - Training progress logs")
    
    print("\n📈 To view Tensorboard logs:")
    print("   tensorboard --logdir ./tensorboard_logs/")

if __name__ == "__main__":
    main()