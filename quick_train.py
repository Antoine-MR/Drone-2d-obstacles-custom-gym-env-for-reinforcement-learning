#!/usr/bin/env python3

import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()
ITERATIONS = 5
TIMESTEPS_PER_ITERATION = 50000

def main():
    print("EUREKA TRAINING - QUICK LAUNCH")
    print("=" * 50)
    print(f"Directory: {BASE_DIR}")
    print(f"Iterations: {ITERATIONS}")
    print(f"Timesteps per iteration: {TIMESTEPS_PER_ITERATION}")
    print(f"Estimated total: ~{(ITERATIONS * 5)}-{(ITERATIONS * 10)} minutes")
    print("=" * 50)
    
    response = input("\nLaunch complete training? (y/N): ").strip().lower()
    
    if response != 'y':
        print("Training cancelled")
        return
    
    print("\nLaunching training...")
    
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
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()