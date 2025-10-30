import gymnasium as gym
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Add paths for imports
sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent / "eureka/envs/drone_2d"))

from stable_baselines3 import PPO
import cv2

def evaluate_trained_model(model_path=None, num_episodes=10, render=True):
    """Evaluate the trained model and compare with baseline"""
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    # Find model if not specified
    if not model_path:
        models_dir = Path("./models/")
        if not models_dir.exists():
            print("❌ No models directory found. Train a model first.")
            return
        
        model_files = list(models_dir.glob("*.zip"))
        if not model_files:
            print("❌ No trained models found")
            return
        
        # Use the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"📁 Using model: {model_path}")
    
    # Import environments
    try:
        from drone_2d_obs import Drone2DCustom  # Eureka optimized
        # Also import original for comparison
        sys.path.append("../../../../drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/")
        from drone_2d_env import Drone2DEnv  # Original
        print("✅ Successfully imported both environments")
    except ImportError as e:
        print(f"❌ Error importing environments: {e}")
        return
    
    # Load trained model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Evaluation function
    def run_evaluation(env_class, env_name, model, num_episodes):
        """Run evaluation on given environment"""
        results = {
            'rewards': [],
            'episode_lengths': [],
            'success_rate': 0,
            'trajectories': []
        }
        
        successes = 0
        
        for episode in range(num_episodes):
            env = env_class()
            obs, _ = env.reset()
            
            episode_reward = 0
            episode_length = 0
            trajectory = {'positions': [], 'targets': [], 'rewards': []}
            
            done = False
            while not done:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                # Store trajectory data
                if hasattr(env, 'drone'):
                    pos = env.drone.body.position
                    target = getattr(env, 'target_position', [0, 0])
                    trajectory['positions'].append([pos.x, pos.y])
                    trajectory['targets'].append(target)
                    trajectory['rewards'].append(reward)
                
                # Check success (reached target)
                if hasattr(info, 'get') and info.get('is_success', False):
                    successes += 1
                
                # Render if requested
                if render and episode == 0:  # Only render first episode
                    try:
                        env.render()
                        time.sleep(0.01)
                    except:
                        pass
            
            env.close()
            
            results['rewards'].append(episode_reward)
            results['episode_lengths'].append(episode_length)
            results['trajectories'].append(trajectory)
            
            print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}")
        
        results['success_rate'] = successes / num_episodes
        return results
    
    print(f"\n🧪 Evaluating on {num_episodes} episodes...")
    
    # Evaluate on optimized environment
    print("\n1️⃣ Evaluating with EUREKA-OPTIMIZED reward...")
    optimized_results = run_evaluation(Drone2DCustom, "Optimized", model, num_episodes)
    
    # Evaluate on original environment for comparison
    print("\n2️⃣ Evaluating with ORIGINAL reward...")
    original_results = run_evaluation(Drone2DEnv, "Original", model, num_episodes)
    
    # Compare results
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    print("🎯 EUREKA-OPTIMIZED Environment:")
    print(f"   Average Reward: {np.mean(optimized_results['rewards']):.2f} ± {np.std(optimized_results['rewards']):.2f}")
    print(f"   Average Episode Length: {np.mean(optimized_results['episode_lengths']):.1f}")
    print(f"   Success Rate: {optimized_results['success_rate']*100:.1f}%")
    
    print("\n🎯 ORIGINAL Environment:")
    print(f"   Average Reward: {np.mean(original_results['rewards']):.2f} ± {np.std(original_results['rewards']):.2f}")
    print(f"   Average Episode Length: {np.mean(original_results['episode_lengths']):.1f}")
    print(f"   Success Rate: {original_results['success_rate']*100:.1f}%")
    
    # Calculate improvement
    reward_improvement = np.mean(optimized_results['rewards']) - np.mean(original_results['rewards'])
    print(f"\n📈 IMPROVEMENT: {reward_improvement:+.2f} reward points")
    
    # Plot results
    plot_evaluation_results(optimized_results, original_results)
    
    return optimized_results, original_results

def plot_evaluation_results(optimized_results, original_results):
    """Create visualization of evaluation results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Eureka Optimization Results', fontsize=16)
    
    # Reward comparison
    axes[0,0].boxplot([original_results['rewards'], optimized_results['rewards']], 
                      labels=['Original', 'Eureka-Optimized'])
    axes[0,0].set_title('Reward Distribution')
    axes[0,0].set_ylabel('Episode Reward')
    
    # Episode length comparison  
    axes[0,1].boxplot([original_results['episode_lengths'], optimized_results['episode_lengths']], 
                      labels=['Original', 'Eureka-Optimized'])
    axes[0,1].set_title('Episode Length Distribution')
    axes[0,1].set_ylabel('Steps')
    
    # Reward over episodes
    axes[1,0].plot(original_results['rewards'], 'b-', label='Original', alpha=0.7)
    axes[1,0].plot(optimized_results['rewards'], 'r-', label='Eureka-Optimized', alpha=0.7)
    axes[1,0].set_title('Rewards Over Episodes')
    axes[1,0].set_xlabel('Episode')
    axes[1,0].set_ylabel('Reward')
    axes[1,0].legend()
    
    # Success rates
    success_data = [original_results['success_rate']*100, optimized_results['success_rate']*100]
    axes[1,1].bar(['Original', 'Eureka-Optimized'], success_data, 
                  color=['blue', 'red'], alpha=0.7)
    axes[1,1].set_title('Success Rate Comparison')
    axes[1,1].set_ylabel('Success Rate (%)')
    axes[1,1].set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "evaluation_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Evaluation plot saved: {plot_path}")
    
    try:
        plt.show()
    except:
        print("📊 Plot saved but cannot display (no GUI available)")

def create_trajectory_animation(results):
    """Create animation of drone trajectories"""
    # This would create a video of the drone's path
    # Implementation depends on your specific visualization needs
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    optimized_results, original_results = evaluate_trained_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render
    )