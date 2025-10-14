# Eureka + Drone 2D Integration Guide

## Objective
Integrate custom 2D drone environment with Eureka for automatic reward function optimization via LLM.

## Architecture

### 1. Eureka Environment Configuration
- **`eureka/cfg/env/drone_2d.yaml`** : Drone environment configuration for Eureka
- **`eureka/envs/drone_2d/`** : Drone environment package adapted for Eureka
  - `drone_2d_env.py` : Main wrapper and reward functions
  - `drone_2d_temp.py` : Reward function template for optimization
  - `drone_2d_obs.py` : Environment definition and observations

### 2. RL-Baselines3-Zoo Integration
- **`rl-baselines3-zoo/rl_zoo3/drone_2d_custom.py`** : Custom drone environment
- **`rl-baselines3-zoo/hyperparams/ppo.yml`** : Hyperparameters for Drone2D-v0
- **`rl-baselines3-zoo/rl_zoo3/import_envs.py`** : Environment registration

### 3. Eureka Configuration
- **`eureka/cfg/config.yaml`** : Modified to use drone_2d by default
- **`eureka/eureka.py`** : Adaptations to support different environments

## State and Action Spaces

### Observations (8D)
```
obs[0]: velocity_x     (normalized X velocity, -1 to 1)
obs[1]: velocity_y     (normalized Y velocity, -1 to 1)  
obs[2]: angular_vel    (normalized angular velocity, -1 to 1)
obs[3]: angle          (angle du drone normalisé, -1 à 1)
obs[4]: distance_x     (distance X à la cible normalisée, -1 à 1)
obs[5]: distance_y     (distance Y à la cible normalisée, -1 à 1)
obs[6]: pos_x          (position X absolue normalisée, -1 à 1)
obs[7]: pos_y          (position Y absolue normalisée, -1 à 1)
```

### Actions (2D)
```
action[0]: left_motor  (force moteur gauche, -1 à 1)  
action[1]: right_motor (force moteur droite, -1 à 1)
```

## 🎯 Fonction de Reward de Base
La fonction initiale optimise :
- **Distance à la cible** : Récompense inversement proportionnelle à la distance
- **Stabilité** : Pénalise les vitesses angulaires et angles élevés  
- **Contrôle fluide** : Pénalise les actions brusques
- **Pénalité terminale** : Malus important si le drone sort des limites

## 🚀 Comment utiliser

### 1. Pré-requis
```bash
# Installer Ollama et llama3.1
ollama pull llama3.1

# Installer les dépendances
pip install --user ollama hydra-core omegaconf
```

### 2. Lancement simple
```bash
cd Eureka_for_carracing-master/Eureka_for_carracing-master
python launch_drone_eureka.py
```

### 3. Lancement avancé
```bash
python eureka/eureka.py \
  env=drone_2d \
  iteration=10 \
  sample=3 \
  max_iterations=3000 \
  model=llama3.1 \
  suffix=drone_test
```

### 4. Test de l'environnement
```bash
python test_drone_env.py
```

## 📊 Hyperparamètres PPO pour Drone2D
```yaml
Drone2D-v0:
  n_envs: 4
  n_timesteps: 3000
  policy: 'MlpPolicy'
  n_steps: 256
  batch_size: 64
  learning_rate: 3e-4
  # ... voir ppo.yml pour la configuration complète
```

## 🔄 Processus Eureka
1. **Génération** : Le LLM génère des nouvelles fonctions de reward
2. **Entraînement** : PPO entraîne un agent avec chaque fonction
3. **Évaluation** : Test de performance de chaque agent
4. **Sélection** : Garde les meilleures fonctions pour la prochaine itération
5. **Itération** : Répète le processus pour améliorer progressivement

## 📈 Fichiers de sortie
- **`rl_zoo3/drone_2d_custom.py`** : Fonction de reward optimale finale
- **`outputs/`** : Logs et résultats d'entraînement par itération
- **Répertoires temporaires** : Agents et codes générés par itération

## 🎛️ Paramètres d'optimisation
- `iteration` : Nombre d'itérations Eureka (défaut: 10)
- `sample` : Nombre d'échantillons de reward par itération (défaut: 3) 
- `max_iterations` : Nombre d'étapes d'entraînement RL (défaut: 3000)
- `temperature` : Créativité du LLM (défaut: 1.0)

## 🐛 Troubleshooting

### Erreur d'import du drone
```bash
pip install --user -e /path/to/drone_2d_custom_gym_env_package
```

### Ollama non disponible
```bash
ollama serve  # Dans un terminal séparé
ollama pull llama3.1
```

### Problème de chemins
Vérifiez que tous les `sys.path.insert()` pointent vers le bon répertoire du package drone.

## ✅ Status
- ✅ Environnement drone intégré
- ✅ Configuration Eureka adaptée  
- ✅ Hyperparamètres PPO configurés
- ✅ Tests fonctionnels passés
- ✅ Fichiers CarRacing préservés
- 🔄 Prêt pour l'optimisation Eureka !

---

Le système est maintenant configuré pour optimiser automatiquement la fonction de reward du drone via Eureka tout en préservant l'environnement CarRacing existant.