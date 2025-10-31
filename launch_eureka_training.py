#!/usr/bin/env python3
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
ITERATIONS = 5                   # Nombre d'itérations Eureka
TIMESTEPS_PER_ITERATION = 5000    # Timesteps par itération
N_ENVS = 4                        # Environnements parallèles
LEARNING_RATE = 3e-4              # Taux d'apprentissage PPO
BATCH_SIZE = 64                   # Taille des batchs
N_EPOCHS = 10                     # Époques par update PPO
CHECKPOINT_FREQ = 10000           # Fréquence de sauvegarde
N_EVAL_EPISODES = 10              # Épisodes d'évaluation
# ===================================================


class EurekaTrainingManager:
    """Gestionnaire d'entraînement Eureka avec configuration personnalisable"""
    
    def __init__(self, base_dir, config):
        self.base_dir = Path(base_dir)
        self.iterations = config['iterations']
        self.timesteps_per_iteration = config['timesteps_per_iteration']
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / "training_sessions" / f"eureka_session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        self.config = {
            'training': {
                'timesteps': config['timesteps_per_iteration'],
                'n_envs': config['n_envs'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'n_epochs': config['n_epochs']
            },
            'checkpoints': {
                'save_freq': config['checkpoint_freq'],
                'n_eval_episodes': config['n_eval_episodes']
            }
        }
        
        self.interrupted = False
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'iterations': [],
            'status': 'running',
            'config': config
        }
        
    def setup_logging(self):
        log_file = self.session_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def signal_handler(self, signum, frame):
        print("\n" + "="*60)
        print("INTERRUPTION DETECTED!")
        print(f"Session saved: {self.session_dir}")
        print("="*60)
        self.interrupted = True
        self.results['status'] = 'interrupted'
        self.save_results()
        sys.exit(0)
        
    def run_eureka_iteration(self, iteration):
        print(f"\nEUREKA ITERATION {iteration}")
        print("-" * 40)
        
        eureka_dir = self.base_dir / "Eureka_for_carracing-master" / "Eureka_for_carracing-master"
        original_cwd = os.getcwd()
        
        try:
            os.chdir(eureka_dir)
            
            cmd = [
                sys.executable, "eureka/eureka.py",
                "env=drone_2d",
                "model=llama3.1:8b",
                f"iteration=1",
                "sample=2",
                "max_iterations=1000",
                f"suffix=iter{iteration}"
            ]
            
            print(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Eureka iteration completed successfully")
                
                self.save_reward_function(iteration)
                return True
            else:
                print(f"Eureka iteration failed with code {result.returncode}")
                print(f"Error: {result.stderr}")
                return False
                
        finally:
            os.chdir(original_cwd)
            
    def save_reward_function(self, iteration):
        eureka_dir = self.base_dir / "Eureka_for_carracing-master" / "Eureka_for_carracing-master"
        
        reward_files = list(eureka_dir.glob("**/env_iter*_response*.py"))
        
        import time
        recent_files = [f for f in reward_files if (time.time() - f.stat().st_mtime) < 3600]
        
        if recent_files:
            latest_reward = max(recent_files, key=lambda x: x.stat().st_mtime)
            
            reward_dir = self.session_dir / "reward_functions"
            reward_dir.mkdir(exist_ok=True)
            
            dest_file = reward_dir / f"reward_iter{iteration}.py"
            shutil.copy2(latest_reward, dest_file)
            
            print(f"Reward function saved: {dest_file}")
            print(f"  Source: {latest_reward.name}")
            return str(dest_file)
        else:
            print("No reward function file found")
            return None
            
    def update_environment_reward(self, reward_file_path):
        try:
            with open(reward_file_path, 'r') as f:
                reward_code = f.read()
            
            import re
            reward_code = re.sub(r':\s*Union\[np\.ndarray,\s*int\]', '', reward_code)
            reward_code = re.sub(r':\s*Tuple\[bool,\s*bool,\s*float\]', '', reward_code)
            reward_code = re.sub(r'->\s*Tuple\[bool,\s*bool,\s*float\]', '', reward_code)
            reward_code = re.sub(r'action:\s*,', 'action,', reward_code)
            reward_code = re.sub(r'\)\s*:', '):', reward_code)
            
            if "def compute_reward(" in reward_code:
                start_idx = reward_code.find("def compute_reward(")
                remaining = reward_code[start_idx:]
                lines = remaining.split('\n')
                
                func_lines = [lines[0]]
                indent_level = None
                
                for line in lines[1:]:
                    if line.strip() == "":
                        func_lines.append(line)
                        continue
                    
                    if indent_level is None and line.strip():
                        indent_level = len(line) - len(line.lstrip())
                    
                    if line.strip() and indent_level and (len(line) - len(line.lstrip())) <= indent_level and not line.startswith(' '):
                        if line.strip().startswith('def ') or line.strip().startswith('class '):
                            break
                    
                    func_lines.append(line)
                
                compute_reward_func = '\n'.join(func_lines)
                
                env_file = (self.base_dir / "Eureka_for_carracing-master" / "Eureka_for_carracing-master" / 
                           "eureka" / "envs" / "drone_2d" / "drone_2d_obs.py")
                
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        content = f.read()
                    
                    if "def compute_reward(" in content:
                        start = content.find("def compute_reward(")
                        if start != -1:
                            lines = content[start:].split('\n')
                            end_line = 1
                            indent_level = None
                            
                            for i, line in enumerate(lines[1:], 1):
                                if line.strip() == "":
                                    continue
                                
                                if indent_level is None and line.strip():
                                    indent_level = len(line) - len(line.lstrip())
                                
                                if line.strip() and indent_level and (len(line) - len(line.lstrip())) <= indent_level and not line.startswith(' '):
                                    if line.strip().startswith('def ') or line.strip().startswith('class '):
                                        end_line = i
                                        break
                            
                            before = content[:start]
                            after_lines = content[start:].split('\n')[end_line:]
                            after = '\n'.join(after_lines) if after_lines else ''
                            
                            new_content = before + compute_reward_func + '\n' + after
                        else:
                            new_content = content + '\n\n' + compute_reward_func
                    else:
                        new_content = content + '\n\n' + compute_reward_func
                    
                    with open(env_file, 'w') as f:
                        f.write(new_content)
                    
                    print("Environment reward function updated")
                    return True
                else:
                    print("Environment file not found")
                    return False
            else:
                print("No compute_reward function found in reward file")
                return False
                
        except Exception as e:
            print(f"Error updating environment: {e}")
            return False
            
    def train_agent_with_reward(self, reward_file_path, iteration):
        try:
            if not self.update_environment_reward(reward_file_path):
                raise Exception("Failed to update reward function")
            
            eureka_dir = self.base_dir / "Eureka_for_carracing-master" / "Eureka_for_carracing-master"
            original_cwd = os.getcwd()
            
            try:
                os.chdir(eureka_dir)
                
                if str(eureka_dir) not in sys.path:
                    sys.path.insert(0, str(eureka_dir))
                
                from stable_baselines3 import PPO
                from stable_baselines3.common.env_util import make_vec_env
                from stable_baselines3.common.callbacks import CheckpointCallback
                import gymnasium as gym
                
                env_path = eureka_dir / "eureka" / "envs" / "drone_2d"
                if str(env_path) not in sys.path:
                    sys.path.insert(0, str(env_path))
                
                import importlib
                if 'drone_2d_obs' in sys.modules:
                    importlib.reload(sys.modules['drone_2d_obs'])
                
                from drone_2d_obs import Drone2DCustom
                
                def make_env():
                    return Drone2DCustom()
                
                env = make_vec_env(make_env, n_envs=self.config['training']['n_envs'])
                
                # Détection automatique du type de policy
                policy_type = "MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "MlpPolicy"
                
                model = PPO(
                    policy_type, 
                    env, 
                    learning_rate=self.config['training']['learning_rate'],
                    batch_size=self.config['training']['batch_size'],
                    n_epochs=self.config['training']['n_epochs'],
                    verbose=1,
                    tensorboard_log=str(self.session_dir / "tensorboard")
                )
                
                checkpoint_dir = self.session_dir / "checkpoints" / f"iteration_{iteration}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_callback = CheckpointCallback(
                    save_freq=self.config['checkpoints']['save_freq'],
                    save_path=str(checkpoint_dir),
                    name_prefix=f"model_iter{iteration}"
                )
                
                print(f"\nTraining PPO model for {self.timesteps_per_iteration} timesteps")
                model.learn(
                    total_timesteps=self.timesteps_per_iteration,
                    callback=checkpoint_callback,
                    tb_log_name=f"iteration_{iteration}"
                )
                
                final_model_path = self.session_dir / "models" / f"model_iter{iteration}_final.zip"
                final_model_path.parent.mkdir(exist_ok=True)
                model.save(str(final_model_path))
                
                print(f"Model saved: {final_model_path}")
                
                evaluation_results = self.evaluate_model(model, iteration)
                
                env.close()
                return evaluation_results
                
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def evaluate_model(self, model, iteration):
        try:
            print(f"\nEvaluating model for iteration {iteration}")
            
            def make_env():
                from drone_2d_obs import Drone2DCustom
                return Drone2DCustom()
            
            from stable_baselines3.common.env_util import make_vec_env
            import numpy as np
            
            eval_env = make_vec_env(make_env, n_envs=1)
            
            n_eval_episodes = self.config['checkpoints']['n_eval_episodes']
            episode_rewards = []
            successes = 0
            crashes = 0
            distances_to_target = []
            episode_lengths = []
            
            for episode in range(n_eval_episodes):
                obs = eval_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]
                    
                    if done[0]:
                        # Extract task metrics from info
                        if len(info) > 0 and isinstance(info[0], dict):
                            episode_info = info[0]
                            if 'success' in episode_info:
                                successes += int(episode_info['success'])
                            if 'crash' in episode_info:
                                crashes += int(episode_info['crash'])
                            if 'distance_to_target' in episode_info:
                                distances_to_target.append(float(episode_info['distance_to_target']))
                            if 'episode_length' in episode_info:
                                episode_lengths.append(int(episode_info['episode_length']))
                        break
                
                episode_rewards.append(episode_reward)
                print(f"  Episode {episode + 1}: {episode_reward:.2f}")
            
            eval_env.close()
            
            results = {
                'mean_reward': float(sum(episode_rewards) / len(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'min_reward': float(min(episode_rewards)),
                'max_reward': float(max(episode_rewards)),
                'episode_rewards': episode_rewards,
                'success_rate': float(successes / n_eval_episodes),
                'crash_rate': float(crashes / n_eval_episodes),
                'avg_distance_to_target': float(np.mean(distances_to_target)) if distances_to_target else None,
                'avg_episode_length': float(np.mean(episode_lengths)) if episode_lengths else None
            }
            
            print(f"Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
            print(f"Success rate: {results['success_rate']*100:.1f}% | Crash rate: {results['crash_rate']*100:.1f}%")
            if results['avg_distance_to_target'] is not None:
                print(f"Avg distance to target: {results['avg_distance_to_target']:.2f}")
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
            
    def run_training_session(self):
        try:
            print("\n" + "=" * 60)
            print("EUREKA TRAINING SESSION")
            print("=" * 60)
            print(f"Session ID: {self.session_id}")
            print(f"Session directory: {self.session_dir}")
            print(f"Iterations: {self.iterations}")
            print(f"Timesteps per iteration: {self.timesteps_per_iteration}")
            print(f"Parallel environments: {self.config['training']['n_envs']}")
            print(f"Learning rate: {self.config['training']['learning_rate']}")
            print("\nPress Ctrl+C anytime to stop gracefully")
            print("=" * 60)
            
            for iteration in range(1, self.iterations + 1):
                if self.interrupted:
                    break
                    
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration}/{self.iterations}")
                print("="*60)
                
                iteration_start_time = time.time()
                iteration_result = {
                    'iteration': iteration,
                    'start_time': datetime.now().isoformat(),
                    'eureka_success': False,
                    'training_success': False,
                    'evaluation_results': None
                }
                
                eureka_success = self.run_eureka_iteration(iteration)
                iteration_result['eureka_success'] = eureka_success
                
                if eureka_success and not self.interrupted:
                    print(f"\nTRAINING AGENT - ITERATION {iteration}")
                    print("-" * 40)
                    
                    reward_file = self.session_dir / "reward_functions" / f"reward_iter{iteration}.py"
                    if reward_file.exists():
                        evaluation_results = self.train_agent_with_reward(str(reward_file), iteration)
                        iteration_result['training_success'] = evaluation_results is not None
                        iteration_result['evaluation_results'] = evaluation_results
                    else:
                        print("Skipping training: reward file not found")
                else:
                    print(f"Skipping training for iteration {iteration}")
                
                iteration_result['end_time'] = datetime.now().isoformat()
                iteration_result['duration'] = time.time() - iteration_start_time
                
                self.results['iterations'].append(iteration_result)
                self.save_results()
                
                # Afficher le résumé de l'itération
                print("\n" + "=" * 60)
                print(f"ITERATION {iteration}/{self.iterations} - SUMMARY")
                print("=" * 60)
                print(f"Duration: {iteration_result['duration']:.2f} seconds ({iteration_result['duration']/60:.1f} minutes)")
                print(f"Eureka: {'Success' if iteration_result['eureka_success'] else 'Failed'}")
                print(f"Training: {'Success' if iteration_result['training_success'] else 'Failed'}")
                
                if iteration_result['evaluation_results']:
                    eval_res = iteration_result['evaluation_results']
                    print("\nEVALUATION RESULTS:")
                    print(f"  Mean reward: {eval_res['mean_reward']:.2f} +/- {eval_res['std_reward']:.2f}")
                    print(f"  Min reward: {eval_res['min_reward']:.2f}")
                    print(f"  Max reward: {eval_res['max_reward']:.2f}")
                    print(f"  Success rate: {eval_res['success_rate']*100:.1f}%")
                    print(f"  Crash rate: {eval_res['crash_rate']*100:.1f}%")
                    if eval_res['avg_distance_to_target'] is not None:
                        print(f"  Avg distance to target: {eval_res['avg_distance_to_target']:.2f}")
                    if eval_res['avg_episode_length'] is not None:
                        print(f"  Avg episode length: {eval_res['avg_episode_length']:.1f} steps")
                    print(f"  Episodes: {len(eval_res['episode_rewards'])}")
                else:
                    print("\nNo evaluation results available")
                
                print("=" * 60 + "\n")
                
            self.results['end_time'] = datetime.now().isoformat()
            self.results['status'] = 'completed' if not self.interrupted else 'interrupted'
            self.save_results()
            
            # Afficher le résumé final
            print("\n" + "=" * 60)
            print("TRAINING SESSION COMPLETED")
            print("=" * 60)
            print(f"Status: {self.results['status']}")
            print(f"Session ID: {self.session_id}")
            print(f"Directory: {self.session_dir}")
            print(f"\nITERATIONS SUMMARY:")
            
            successful_iterations = []
            best_success_rate = -1
            best_success_iter = None
            
            for iter_res in self.results['iterations']:
                iter_num = iter_res['iteration']
                success = "[OK]" if iter_res['training_success'] else "[FAIL]"
                
                if iter_res['evaluation_results']:
                    eval_res = iter_res['evaluation_results']
                    mean_reward = eval_res['mean_reward']
                    success_rate = eval_res.get('success_rate', 0)
                    
                    successful_iterations.append((iter_num, mean_reward, success_rate))
                    
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_success_iter = (iter_num, eval_res)
                    
                    print(f"  Iteration {iter_num}: {success} Reward={mean_reward:.2f} | Success={success_rate*100:.1f}%")
                else:
                    print(f"  Iteration {iter_num}: {success} No results")
            
            if successful_iterations:
                print(f"\nBEST ITERATION (by success rate):")
                if best_success_iter:
                    best_iter, best_eval = best_success_iter
                    print(f"  Iteration: #{best_iter}")
                    print(f"  Success rate: {best_eval['success_rate']*100:.1f}%")
                    print(f"  Mean reward: {best_eval['mean_reward']:.2f}")
                    print(f"  Crash rate: {best_eval['crash_rate']*100:.1f}%")
                    if best_eval['avg_distance_to_target'] is not None:
                        print(f"  Avg distance to target: {best_eval['avg_distance_to_target']:.2f}")
                    print(f"  Model saved: models/model_iter{best_iter}_final.zip")
                else:
                    best_iter, best_reward, _ = max(successful_iterations, key=lambda x: x[1])
                    print(f"  Iteration: #{best_iter} (by reward)")
                    print(f"  Mean reward: {best_reward:.2f}")
                    print(f"  Model saved: models/model_iter{best_iter}_final.zip")
            
            print("\n" + "=" * 60)
            print(f"All results saved to:\n  {self.session_dir}")
            print("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Training session failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.save_results()
            
    def save_results(self):
        results_file = self.session_dir / "training_summary.json"
        
        # Convertir les types numpy en types Python natifs pour JSON
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_to_save = convert_numpy_types(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Training summary saved: {results_file}")


def main():
    """Point d'entrée principal"""
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Configuration à partir des variables globales
    config = {
        'iterations': ITERATIONS,
        'timesteps_per_iteration': TIMESTEPS_PER_ITERATION,
        'n_envs': N_ENVS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'checkpoint_freq': CHECKPOINT_FREQ,
        'n_eval_episodes': N_EVAL_EPISODES
    }
    
    print("=" * 60)
    print("EUREKA DRONE TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Eureka iterations: {config['iterations']}")
    print(f"  Timesteps/iteration: {config['timesteps_per_iteration']:,}")
    print(f"  Parallel environments: {config['n_envs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  PPO epochs: {config['n_epochs']}")
    print(f"  Eval episodes: {config['n_eval_episodes']}")
    
    estimated_time_min = config['iterations'] * 5
    estimated_time_max = config['iterations'] * 10
    print(f"\nEstimated time: ~{estimated_time_min}-{estimated_time_max} minutes")
    print(f"Working directory: {BASE_DIR}")
    print("\n" + "=" * 60)
    
    try:
        trainer = EurekaTrainingManager(base_dir=BASE_DIR, config=config)
        trainer.run_training_session()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
