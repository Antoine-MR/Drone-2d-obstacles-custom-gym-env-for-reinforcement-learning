#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de debug pour identifier le problème avec l'API reset() de l'environnement
"""

import sys
import os
from pathlib import Path

# Fix encoding pour Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("DEBUG SCRIPT - Analyse du probleme reset()")
print("=" * 70)

# Test 1: Importer l'environnement original
print("\n[TEST 1] Import de l'environnement Drone original")
print("-" * 70)
try:
    import drone_2d_custom_gym_env
    import gym
    
    print("[OK] Package importe avec succes")
    
    # Créer l'environnement
    env = gym.make('drone-2d-custom-v0')
    print(f"✓ Environnement créé : {type(env)}")
    
    # Tester reset()
    print("\n[Appel] env.reset()")
    result = env.reset()
    print(f"✓ reset() retourne {len(result) if isinstance(result, tuple) else 1} valeur(s)")
    print(f"  Type de retour : {type(result)}")
    
    if isinstance(result, tuple):
        for i, val in enumerate(result):
            print(f"  - Valeur {i+1} : type={type(val)}, shape={getattr(val, 'shape', 'N/A')}")
    else:
        print(f"  - Valeur unique : type={type(result)}, shape={getattr(result, 'shape', 'N/A')}")
    
    env.close()
    
except Exception as e:
    print(f"✗ Erreur : {e}")
    import traceback
    traceback.print_exc()

# Test 2: Vérifier la signature de reset()
print("\n\n[TEST 2] Analyse de la signature reset()")
print("-" * 70)
try:
    from drone_2d_custom_gym_env.drone_2d_env import Drone2dEnv
    import inspect
    
    sig = inspect.signature(Drone2dEnv.reset)
    print(f"Signature : {sig}")
    
    # Lire le code source
    source = inspect.getsource(Drone2dEnv.reset)
    print("\nCode source de reset() :")
    print("-" * 70)
    print(source)
    
except Exception as e:
    print(f"✗ Erreur : {e}")

# Test 3: Tester avec Stable-Baselines3
print("\n\n[TEST 3] Test avec Stable-Baselines3 VecEnv")
print("-" * 70)
try:
    from stable_baselines3.common.env_util import make_vec_env
    import drone_2d_custom_gym_env
    
    def make_env():
        return gym.make('drone-2d-custom-v0')
    
    print("✓ Création d'un VecEnv...")
    vec_env = make_vec_env(make_env, n_envs=1)
    
    print("[Appel] vec_env.reset()")
    obs = vec_env.reset()
    print(f"✓ reset() fonctionne avec VecEnv")
    print(f"  Observation shape : {obs.shape}")
    
    vec_env.close()
    
except Exception as e:
    print(f"✗ Erreur avec VecEnv : {e}")
    import traceback
    traceback.print_exc()

# Test 4: Vérifier l'API Gymnasium vs Gym
print("\n\n[TEST 4] Vérification des versions d'API")
print("-" * 70)
try:
    import gym
    print(f"gym version : {gym.__version__}")
    
    try:
        import gymnasium
        print(f"gymnasium version : {gymnasium.__version__}")
    except ImportError:
        print("gymnasium non installé")
    
    from stable_baselines3 import __version__ as sb3_version
    print(f"stable-baselines3 version : {sb3_version}")
    
except Exception as e:
    print(f"✗ Erreur : {e}")

# Test 5: Analyser le fichier drone_2d_env.py
print("\n\n[TEST 5] Analyse du fichier drone_2d_env.py")
print("-" * 70)
try:
    drone_env_path = Path("drone_2d_custom_gym_env_package/drone_2d_custom_gym_env/drone_2d_env.py")
    if drone_env_path.exists():
        with open(drone_env_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Trouver la méthode reset
        import re
        reset_match = re.search(r'def reset\(.*?\):(.*?)(?=\n    def |\nclass |\Z)', content, re.DOTALL)
        
        if reset_match:
            reset_code = reset_match.group(0)
            
            # Chercher les return statements
            returns = re.findall(r'return (.+)', reset_code)
            
            print(f"✓ Fichier trouvé : {drone_env_path}")
            print(f"  Nombre de 'return' trouvés : {len(returns)}")
            
            for i, ret in enumerate(returns, 1):
                print(f"\n  Return #{i} :")
                print(f"    {ret}")
                
                # Compter les éléments retournés
                elements = [x.strip() for x in ret.split(',')]
                print(f"    → {len(elements)} élément(s) : {elements}")
        else:
            print("⚠ Méthode reset() non trouvée dans le fichier")
    else:
        print(f"✗ Fichier non trouvé : {drone_env_path}")
        
except Exception as e:
    print(f"✗ Erreur : {e}")
    import traceback
    traceback.print_exc()

# Résumé et recommandations
print("\n\n" + "=" * 70)
print("RÉSUMÉ ET DIAGNOSTIC")
print("=" * 70)
print("""
Le problème 'too many values to unpack (expected 2)' indique que :

1. Stable-Baselines3 attend : reset() → (observation, info)
2. Votre environnement retourne probablement : reset() → (obs, reward, done, info)
   OU : reset() → observation seulement

SOLUTIONS POSSIBLES :
""")
print("=" * 70)
