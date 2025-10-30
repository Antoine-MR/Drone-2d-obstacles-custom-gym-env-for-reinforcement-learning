#!/usr/bin/env python3

import json
import os
import time
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "training_sessions"
        
    def list_sessions(self):
        if not self.sessions_dir.exists():
            print("No session found")
            return []
            
        sessions = []
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('eureka_session_'):
                sessions.append(session_dir)
        
        sessions.sort(key=lambda x: x.name, reverse=True)
        return sessions
    
    def get_session_status(self, session_dir):
        summary_file = session_dir / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                return {"error": str(e)}
        return {"status": "no_summary"}
    
    def monitor_active_session(self, session_id=None):
        sessions = self.list_sessions()
        
        if session_id:
            target_session = None
            for session in sessions:
                if session_id in session.name:
                    target_session = session
                    break
            if not target_session:
                print(f"Session {session_id} not found")
                return
        else:
            if not sessions:
                print("No session found")
                return
            target_session = sessions[0]
        
        print(f"Monitoring session: {target_session.name}")
        print("=" * 60)
        
        while True:
            try:
                status = self.get_session_status(target_session)
                
                print(f"\n{datetime.now().strftime('%H:%M:%S')}")
                print(f"Session: {target_session.name}")
                print(f"Status: {status.get('status', 'unknown')}")
                
                if 'iterations' in status:
                    completed_iterations = len(status['iterations'])
                    print(f"Completed iterations: {completed_iterations}")
                    
                    if status['iterations']:
                        last_iter = status['iterations'][-1]
                        print(f"   - Last iteration: {last_iter['iteration']}")
                        print(f"   - Eureka success: {'yes' if last_iter.get('eureka_success') else 'no'}")
                        print(f"   - Training success: {'yes' if last_iter.get('training_success') else 'no'}")
                        
                        if last_iter.get('evaluation_results'):
                            eval_results = last_iter['evaluation_results']
                            print(f"   - Mean reward: {eval_results.get('mean_reward', 'N/A'):.2f}")
                
                reward_dir = target_session / "reward_functions"
                if reward_dir.exists():
                    reward_files = list(reward_dir.glob("*.py"))
                    print(f"Reward functions: {len(reward_files)}")
                
                checkpoints_dir = target_session / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
                    print(f"Checkpoints: {len(checkpoint_dirs)} iterations")
                
                if status.get('status') in ['completed', 'failed', 'interrupted']:
                    print(f"\nSession terminated with status: {status['status']}")
                    break
                
                print("-" * 40)
                time.sleep(10)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def print_session_summary(self, session_dir):
        status = self.get_session_status(session_dir)
        
        print(f"\nSUMMARY - {session_dir.name}")
        print("=" * 50)
        
        if 'error' in status:
            print(f"Error: {status['error']}")
            return
        
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Start: {status.get('start_time', 'N/A')}")
        print(f"End: {status.get('end_time', 'N/A')}")
        
        if 'iterations' in status:
            print(f"Iterations: {len(status['iterations'])}")
            
            for i, iteration in enumerate(status['iterations'], 1):
                print(f"\n  Iteration {i}:")
                print(f"    - Eureka: {'yes' if iteration.get('eureka_success') else 'no'}")
                print(f"    - Training: {'yes' if iteration.get('training_success') else 'no'}")
                
                if iteration.get('evaluation_results'):
                    results = iteration['evaluation_results']
                    print(f"    - Mean reward: {results.get('mean_reward', 0):.2f}")
                    print(f"    - Std reward: {results.get('std_reward', 0):.2f}")
                
                if 'duration' in iteration:
                    duration_min = iteration['duration'] / 60
                    print(f"    - Duration: {duration_min:.1f} min")

def main():
    BASE_DIR = r'C:\Users\antoi\Documents\cours\si5\drl\Drone-2d-obstacles-custom-gym-env-for-reinforcement-learning'
    
    monitor = TrainingMonitor(BASE_DIR)
    
    print("EUREKA TRAINING MONITOR")
    print("=" * 40)
    
    sessions = monitor.list_sessions()
    
    if not sessions:
        print("No training session found")
        return
    
    print(f"Sessions found: {len(sessions)}")
    
    for i, session in enumerate(sessions):
        print(f"\n{i+1}. {session.name}")
        status = monitor.get_session_status(session)
        print(f"   Status: {status.get('status', 'unknown')}")
        
        if 'iterations' in status:
            completed = len(status['iterations'])
            print(f"   Iterations: {completed}")
    
    print("\n" + "=" * 40)
    print("Options:")
    print("1. Monitor most recent session")
    print("2. View summary of all sessions")
    print("3. Monitor specific session")
    
    try:
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            monitor.monitor_active_session()
        elif choice == "2":
            for session in sessions:
                monitor.print_session_summary(session)
        elif choice == "3":
            print("\nAvailable sessions:")
            for i, session in enumerate(sessions):
                print(f"{i+1}. {session.name}")
            
            try:
                idx = int(input("Session number: ")) - 1
                if 0 <= idx < len(sessions):
                    session_id = sessions[idx].name.split('_')[-1]
                    monitor.monitor_active_session(session_id)
                else:
                    print("Invalid number")
            except ValueError:
                print("Invalid number")
        else:
            print("Invalid option")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()