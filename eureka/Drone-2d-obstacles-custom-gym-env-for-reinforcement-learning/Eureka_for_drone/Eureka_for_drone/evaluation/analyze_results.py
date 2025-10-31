import os
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_eureka_results():
    """Analyze Eureka training results and find best reward function"""
    print("📊 ANALYSIS OF EUREKA RESULTS")
    print("="*50)
    
    output_dir = Path("outputs/eureka")
    if not output_dir.exists():
        print("❌ No Eureka results found")
        return None
    
    # Find latest training session
    recent_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], 
                        key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not recent_dirs:
        print("❌ No training sessions found")
        return None
    
    latest_dir = recent_dirs[0]
    print(f"📁 Analyzing session: {latest_dir.name}")
    
    # Parse log file for performance metrics
    log_file = latest_dir / "eureka.log"
    best_rewards = []
    iterations = []
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            for line in f:
                if "Max Success:" in line:
                    try:
                        # Extract reward value from log
                        parts = line.split("Max Success:")
                        if len(parts) > 1:
                            reward_str = parts[1].split(",")[0].strip()
                            reward = float(reward_str)
                            best_rewards.append(reward)
                    except:
                        pass
                
                if "Iteration" in line and "Generating" in line:
                    try:
                        iter_num = int(line.split("Iteration")[1].split(":")[0].strip())
                        iterations.append(iter_num)
                    except:
                        pass
    
    # Find best performing reward functions
    reward_files = list(latest_dir.glob("*_rewardonly.py"))
    env_files = list(latest_dir.glob("env_iter*.py"))
    
    print(f"🎯 Generated {len(reward_files)} reward functions")
    print(f"🔄 Completed {len(env_files)} iterations")
    
    if best_rewards:
        print(f"📈 Best reward achieved: {max(best_rewards):.2f}")
        print(f"📉 Average reward: {np.mean(best_rewards):.2f}")
    
    # Identify best reward function
    best_reward_file = None
    if reward_files:
        # Sort by modification time (latest is likely best)
        best_reward_file = max(reward_files, key=lambda x: x.stat().st_mtime)
        print(f"🏆 Best reward function: {best_reward_file.name}")
    
    return {
        'session_dir': latest_dir,
        'best_reward_file': best_reward_file,
        'reward_history': best_rewards,
        'num_iterations': len(env_files)
    }

def display_best_reward_function(results):
    """Display the content of the best reward function"""
    if not results or not results['best_reward_file']:
        print("❌ No reward function to display")
        return
    
    print("\n" + "="*60)
    print("🏆 BEST REWARD FUNCTION")
    print("="*60)
    
    try:
        with open(results['best_reward_file'], 'r') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Error reading file: {e}")
    
    print("="*60)

if __name__ == "__main__":
    results = analyze_eureka_results()
    if results:
        display_best_reward_function(results)