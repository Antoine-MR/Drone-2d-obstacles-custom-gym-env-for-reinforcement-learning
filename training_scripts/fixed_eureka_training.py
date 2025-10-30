import os
import sys
import json
import time
import signal
import subprocess
import shutil
import logging
from datetime import datetime
from pathlib import Path

class EurekaTrainingManager:
    def __init__(self, base_dir, iterations=5, timesteps_per_iteration=50000):
        self.base_dir = Path(base_dir)
        self.iterations = iterations
        self.timesteps_per_iteration = timesteps_per_iteration
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / "training_sessions" / f"eureka_session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        self.config = {
            'training': {
                'timesteps': timesteps_per_iteration,
                'n_envs': 4,
                'learning_rate': 3e-4,
                'batch_size': 64,
                'n_epochs': 10
            },
            'checkpoints': {
                'save_freq': 10000,
                'n_eval_episodes': 10
            }
        }
        
        self.interrupted = False
        signal.signal(signal.SIGINT, self.signal_handler)
        
        self.results = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'iterations': [],
            'status': 'running'
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
        print(f"Session saved in: {self.session_dir}")
        print("="*60)
        self.interrupted = True
        self.results['status'] = 'interrupted'
        self.save_results()
        
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
        
        reward_files = list(eureka_dir.glob("**/reward_iter*.py"))
        if reward_files:
            latest_reward = max(reward_files, key=lambda x: x.stat().st_mtime)
            
            reward_dir = self.session_dir / "reward_functions"
            reward_dir.mkdir(exist_ok=True)
            
            dest_file = reward_dir / f"reward_iter{iteration}.py"
            shutil.copy2(latest_reward, dest_file)
            
            print(f"Reward function saved: {dest_file}")
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
                from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
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
                
                model = PPO(
                    "MlpPolicy", 
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
                
                print(f"Training PPO model for {self.timesteps_per_iteration} timesteps")
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
            print(f"Evaluating model for iteration {iteration}")
            
            def make_env():
                from drone_2d_obs import Drone2DCustom
                return Drone2DCustom()
            
            from stable_baselines3.common.env_util import make_vec_env
            import numpy as np
            
            eval_env = make_vec_env(make_env, n_envs=1)
            
            n_eval_episodes = self.config['checkpoints']['n_eval_episodes']
            episode_rewards = []
            
            for episode in range(n_eval_episodes):
                obs = eval_env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward[0]
                    
                    if done[0]:
                        break
                
                episode_rewards.append(episode_reward)
                print(f"  Episode {episode + 1}: {episode_reward:.2f}")
            
            eval_env.close()
            
            results = {
                'mean_reward': float(sum(episode_rewards) / len(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'min_reward': float(min(episode_rewards)),
                'max_reward': float(max(episode_rewards)),
                'episode_rewards': episode_rewards
            }
            
            print(f"Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
            return results
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return None
            
    def run_training_session(self):
        try:
            print("EUREKA TRAINING WITH CHECKPOINTS")
            print("=" * 60)
            print(f"Session ID: {self.session_id}")
            print(f"Session directory: {self.session_dir}")
            print(f"Iterations: {self.iterations}")
            print(f"Timesteps per iteration: {self.timesteps_per_iteration}")
            print("Press Ctrl+C anytime to stop gracefully")
            print("=" * 60)
            
            for iteration in range(1, self.iterations + 1):
                if self.interrupted:
                    break
                    
                print(f"\nSTARTING ITERATION {iteration}/{self.iterations}")
                print("=" * 50)
                
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
                
                print(f"Iteration {iteration} completed successfully!")
                
            self.results['end_time'] = datetime.now().isoformat()
            self.results['status'] = 'completed' if not self.interrupted else 'interrupted'
            self.save_results()
            
            print(f"\nTRAINING SESSION COMPLETED!")
            print(f"All results saved in: {self.session_dir}")
            
        except Exception as e:
            self.logger.error(f"Training session failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.save_results()
            
    def save_results(self):
        results_file = self.session_dir / "training_summary.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Training summary saved: {results_file}")

def main():
    import numpy as np
    
    BASE_DIR = Path(__file__).parent.parent.absolute()
    ITERATIONS = 5
    TIMESTEPS_PER_ITERATION = 50000
    
    trainer = EurekaTrainingManager(
        base_dir=BASE_DIR,
        iterations=ITERATIONS,
        timesteps_per_iteration=TIMESTEPS_PER_ITERATION
    )
    
    trainer.run_training_session()

if __name__ == "__main__":
    main()