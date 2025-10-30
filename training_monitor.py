#!/usr/bin/env python3
"""
Script de monitoring des sessions d'entraînement Eureka
"""

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
        """Liste toutes les sessions d'entraînement"""
        if not self.sessions_dir.exists():
            print("❌ Aucune session trouvée")
            return []
            
        sessions = []
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('eureka_session_'):
                sessions.append(session_dir)
        
        sessions.sort(key=lambda x: x.name, reverse=True)
        return sessions
    
    def get_session_status(self, session_dir):
        """Récupère le statut d'une session"""
        summary_file = session_dir / "training_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                return {"error": str(e)}
        return {"status": "no_summary"}
    
    def monitor_active_session(self, session_id=None):
        """Surveille une session active en temps réel"""
        sessions = self.list_sessions()
        
        if session_id:
            target_session = None
            for session in sessions:
                if session_id in session.name:
                    target_session = session
                    break
            if not target_session:
                print(f"❌ Session {session_id} non trouvée")
                return
        else:
            if not sessions:
                print("❌ Aucune session trouvée")
                return
            target_session = sessions[0]  # Plus récente
        
        print(f"👀 Monitoring de la session: {target_session.name}")
        print("=" * 60)
        
        while True:
            try:
                status = self.get_session_status(target_session)
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
                print(f"📁 Session: {target_session.name}")
                print(f"📊 Status: {status.get('status', 'unknown')}")
                
                if 'iterations' in status:
                    completed_iterations = len(status['iterations'])
                    print(f"🔄 Itérations complétées: {completed_iterations}")
                    
                    if status['iterations']:
                        last_iter = status['iterations'][-1]
                        print(f"   - Dernière itération: {last_iter['iteration']}")
                        print(f"   - Eureka success: {'✅' if last_iter.get('eureka_success') else '❌'}")
                        print(f"   - Training success: {'✅' if last_iter.get('training_success') else '❌'}")
                        
                        if last_iter.get('evaluation_results'):
                            eval_results = last_iter['evaluation_results']
                            print(f"   - Mean reward: {eval_results.get('mean_reward', 'N/A'):.2f}")
                
                # Vérifier les fichiers de récompense
                reward_dir = target_session / "reward_functions"
                if reward_dir.exists():
                    reward_files = list(reward_dir.glob("*.py"))
                    print(f"🎯 Fonctions de récompense: {len(reward_files)}")
                
                # Vérifier les checkpoints
                checkpoints_dir = target_session / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
                    print(f"💾 Checkpoints: {len(checkpoint_dirs)} itérations")
                
                if status.get('status') in ['completed', 'failed', 'interrupted']:
                    print(f"\n🏁 Session terminée avec le status: {status['status']}")
                    break
                
                print("-" * 40)
                time.sleep(10)  # Attendre 10 secondes
                
            except KeyboardInterrupt:
                print("\n👋 Monitoring arrêté par l'utilisateur")
                break
            except Exception as e:
                print(f"❌ Erreur de monitoring: {e}")
                time.sleep(5)
    
    def print_session_summary(self, session_dir):
        """Affiche le résumé d'une session"""
        status = self.get_session_status(session_dir)
        
        print(f"\n📋 RÉSUMÉ - {session_dir.name}")
        print("=" * 50)
        
        if 'error' in status:
            print(f"❌ Erreur: {status['error']}")
            return
        
        print(f"📊 Status: {status.get('status', 'unknown')}")
        print(f"🕐 Début: {status.get('start_time', 'N/A')}")
        print(f"🕑 Fin: {status.get('end_time', 'N/A')}")
        
        if 'iterations' in status:
            print(f"🔄 Itérations: {len(status['iterations'])}")
            
            for i, iteration in enumerate(status['iterations'], 1):
                print(f"\n  Itération {i}:")
                print(f"    - Eureka: {'✅' if iteration.get('eureka_success') else '❌'}")
                print(f"    - Training: {'✅' if iteration.get('training_success') else '❌'}")
                
                if iteration.get('evaluation_results'):
                    results = iteration['evaluation_results']
                    print(f"    - Mean reward: {results.get('mean_reward', 0):.2f}")
                    print(f"    - Std reward: {results.get('std_reward', 0):.2f}")
                
                if 'duration' in iteration:
                    duration_min = iteration['duration'] / 60
                    print(f"    - Durée: {duration_min:.1f} min")

def main():
    BASE_DIR = r'C:\Users\antoi\Documents\cours\si5\drl\Drone-2d-obstacles-custom-gym-env-for-reinforcement-learning'
    
    monitor = TrainingMonitor(BASE_DIR)
    
    print("🔍 EUREKA TRAINING MONITOR")
    print("=" * 40)
    
    sessions = monitor.list_sessions()
    
    if not sessions:
        print("❌ Aucune session d'entraînement trouvée")
        return
    
    print(f"📁 Sessions trouvées: {len(sessions)}")
    
    for i, session in enumerate(sessions):
        print(f"\n{i+1}. {session.name}")
        status = monitor.get_session_status(session)
        print(f"   Status: {status.get('status', 'unknown')}")
        
        if 'iterations' in status:
            completed = len(status['iterations'])
            print(f"   Itérations: {completed}")
    
    print("\n" + "=" * 40)
    print("Options:")
    print("1. Monitorer la session la plus récente")
    print("2. Voir le résumé de toutes les sessions")
    print("3. Monitorer une session spécifique")
    
    try:
        choice = input("\nChoisissez une option (1-3): ").strip()
        
        if choice == "1":
            monitor.monitor_active_session()
        elif choice == "2":
            for session in sessions:
                monitor.print_session_summary(session)
        elif choice == "3":
            print("\nSessions disponibles:")
            for i, session in enumerate(sessions):
                print(f"{i+1}. {session.name}")
            
            try:
                idx = int(input("Numéro de session: ")) - 1
                if 0 <= idx < len(sessions):
                    session_id = sessions[idx].name.split('_')[-1]
                    monitor.monitor_active_session(session_id)
                else:
                    print("❌ Numéro invalide")
            except ValueError:
                print("❌ Numéro invalide")
        else:
            print("❌ Option invalide")
            
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")

if __name__ == "__main__":
    main()