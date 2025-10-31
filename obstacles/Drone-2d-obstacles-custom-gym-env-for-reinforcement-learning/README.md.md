# Drone 2D – Obstacles (Gymnasium + SB3)

Ce dépôt contient un environnement Gymnasium personnalisé *drone-2d-custom-v0* et trois scripts principaux pour entraîner, évaluer et comparer des agents PPO avec **Stable-Baselines3**.

> ℹ️ À la racine, un dossier **`Eureka/`** contient également la tentative réalisée avec le framework **Eureka** (expérimentations séparées).

## Table des matières
- [Prérequis](#prérequis)
- [Structure rapide](#structure-rapide)
- [Environnement & paramètres clés](#environnement--paramètres-clés)
- [Scripts principaux](#scripts-principaux)
  - [1) `examples/train.py`](#1-examplestrainpy)
  - [2) `examples/eval.py`](#2-exemplesevalpy)
  - [3) `examples/compare_agents.py`](#3-examplescompare_agentspy)
- [Conseils & bonnes pratiques](#conseils--bonnes-pratiques)

---

## Prérequis
- Python 3.10+ recommandé
- [Gymnasium](https://gymnasium.farama.org/) (API `reset() -> (obs, info)` et `step() -> (obs, reward, terminated, truncated, info)`)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO)
- NumPy, TensorBoard

Installation indicative :
```bash
pip install -U gymnasium stable-baselines3 numpy tensorboard
# si l'environnement est packagé localement, installez-le en editable
# depuis la racine du repo :
pip install -e ./drone_2d_custom_gym_env_package
```

> Les scripts ajoutent également dynamiquement le chemin vers `drone_2d_custom_gym_env_package` via `sys.path.insert(...)`.

---

## Structure rapide
```
├─ Obstacle_experiment/
│  ├─ drone_2d_custom_gym_env_package/
│  │  └─ (code de l’environnement + enregistrement Gymnasium)
│  ├─ examples/
│  │  ├─ train.py
│  │  ├─ eval.py
│  │  └─ compare_agents.py
│  ├─ agents/
│  │  └─ (modèles appris .zip, checkpoints, etc.)
│  └─ (autres fichiers de configuration et logs)
│
└─ Eureka/
   └─ (expérimentations avec le framework Eureka, reward functions générées, sessions d’entraînement, logs, etc.)
```


---

## Environnement & paramètres clés
Les trois scripts créent l’environnement ainsi (valeurs par défaut observées) :
```python
env = gym.make(
    "drone-2d-custom-v0",
    render_sim=False,        # ou True pour afficher la simulation
    render_path=False,       # trace du chemin
    render_shade=False,      # ombrage / zone de visibilité
    shade_distance=70,       # rayon de "shade"
    n_steps=500,             # horizon max par épisode
    n_fall_steps=10,         # pas après lesquels on considère une chute
    change_target=False,     # cible fixe ou changeante
    initial_throw=False,     # lancer initial
    use_obstacles=True,      # activer obstacles
    num_obstacles=3,         # nombre d’obstacles
    fixed_map=True,          # carte fixe
    random_start=True        # position de départ aléatoire
)
```
### Signification rapide
- **render_sim / render_path / render_shade** : contrôle du rendu (coût CPU ↑).
- **shade_distance** : portée de la zone d’ombre/visibilité.
- **n_steps / n_fall_steps** : durée max épisode et dynamique de chute.
- **use_obstacles / num_obstacles** : obstacles activés et leur nombre.
- **fixed_map** : **doit rester à `True`** — cette option garantit une carte figée.
  > ⚠️ Mettre `fixed_map=False` génère des obstacles aléatoires, mais cette option ne fonctionne pas correctement dans la version actuelle de l’environnement.
- **random_start** : position de départ aléatoire (meilleure robustesse).

> Les observations contiennent notamment des indicateurs de limites (ex. `obs[6]`, `obs[7]`), utilisés dans la comparaison pour détecter les sorties de zone.

---

## Scripts principaux

### 1) `examples/train.py`
Entraîne un agent **PPO(MlpPolicy)** avec logs TensorBoard et gestion élégante de l’arrêt manuel.

**Composants clés :**
- **PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")**
- **Callback `KeyboardInterruptCallback`**
  - Sauvegarde un **checkpoint** toutes les **50 000 étapes** : `checkpoint_agent_step_<n>`
  - En cas de `Ctrl+C`, sauve **immédiatement** un modèle : `checkpoint_agent_interrupted_<timestamp>`
  - Attributs importants :
    - `save_path` *(str)* : préfixe des fichiers de sauvegarde (`checkpoint_agent` par défaut).
- **Gestion de signal (`signal.SIGINT`)** : déclenche la sauvegarde finale via le callback.
- **Durée d’entraînement** : `total_timesteps=3_000_000` (modifiable).
- **Sorties** : 
  - Modèle final : `final_agent`
  - Dossiers TensorBoard : `./tensorboard_logs/`

**Lancer l’entraînement :**
```bash
python examples/train.py
# Visualiser ensuite :
tensorboard --logdir=./tensorboard_logs/
# Ouvrir http://localhost:6006
```

**Modifier les hyperparamètres / l’environnement :**
- Éditez directement les arguments de `gym.make(...)` ou les paramètres PPO.
- Changez `save_path`, `total_timesteps`, etc., dans le script.

---

### 2) `examples/eval.py`
Évalue un agent entraîné, en mode continu ou épisode unique, avec rendu activable.

**Options en tête de script :**
```python
continuous_mode = True   # boucle sans fin d’épisodes si True
random_action   = False  # pour jouer aléatoirement (baseline)
use_fixed_map   = True   # répéter sur même carte ou non
render_sim      = True   # afficher la simulation
```

**Chargement du modèle :**
- Parcourt une **liste `model_paths`** et charge le **premier modèle existant** (ex. `../agents/final_agentRandomPosition.zip`, `final_agent.zip`, etc.).
- Définit une **graine aléatoire** basée sur l’horodatage : `model.set_random_seed(...)`.

**Boucle d’évaluation :**
- Action déterministe : `model.predict(obs, deterministic=True)` (sauf si `random_action=True`).
- Comptabilise par épisode : **récompense totale** et **nombre d’étapes**.
- En `continuous_mode=True`, l’environnement est **réinitialisé** automatiquement à chaque fin d’épisode.

**Arrêt & stats :**
- `Ctrl+C` : imprime le nombre d’épisodes terminés + stats de l’épisode en cours.
- Ferme toujours proprement l’environnement (`env.close()`).

**Lancer l’évaluation :**
```bash
python examples/eval.py
```

---

### 3) `examples/compare_agents.py`
Compare **deux agents PPO** sur un protocole identique, répété `NUM_TRIALS` fois.

**Configuration :**
```python
NUM_TRIALS = 30  # essais par agent
RENDER = False   # activer le rendu si besoin

agents = {
    "final_agentRandomPosition": "../agents/final_agentRandomPosition.zip",
    "final_agentWithoutObstacle": "../agents/final_agentWithoutObstacle.zip",
    # ajoutez d’autres agents ici si besoin
}
```
> Si un chemin est manquant, le script le signale et continue.

**Protocole d’évaluation :**
- Environnement **fixe** avec **obstacle central** et **positions de départ aléatoires**.
- Par épisode :
  - action déterministe (`model.predict(..., deterministic=True)`)
  - **Succès** si la distance drone–cible `< 30` (seuil).
  - Sinon, catégorisation de l’échec :
    - **Collision obstacle** via `env.unwrapped.obstacle_manager.check_collision_with_drone(...)`
    - **Sortie de limites** via flags d’observation (ex. `abs(obs[6]) == 1` ou `abs(obs[7]) == 1`)
    - **Timeout** si `episode_steps >= 500` et la distance reste ≥ 30.
- Statistiques par agent :
  - `successes`, `collisions_obstacle`, `collisions_boundary`, `timeouts`
  - `avg_reward ± std_reward`, `avg_steps ± std_steps`
  - `success_rate = successes / NUM_TRIALS`

**Sortie & interprétation :**
- Tableau console détaillé par agent, puis **détermination du “gagnant”** via un score agrégé (pondération succès/collisions/rapidité).  
- Imprime une **analyse** : qui réussit plus souvent, évite mieux les obstacles, va plus vite, etc.

**Lancer la comparaison :**
```bash
python examples/compare_agents.py
```
---

## Agents

Il y a 3 agents entrainées dans le dossier ```agents/``` : 
- ```agent_finalWithoutObstacle```: l'agent entrainé sans obstacles et des positions aléatoires
- ```agent_finalWithObstacle```: l'agent entrainé avec un obstacle et une position fixe
- ```agent_finalRandomPosition```: l'agent entrainé avec un obstacle et des positions aléatoires

Vous pouvez utilisez ceux ci dans les variables de eval.py ou compare_agents.py pour tester un petit les agents entraînés durant le projet.

---

## Dossier **Eureka/**

Un dossier **`Eureka/`** est présent à la racine du projet.  
Il contient notre tentative d’utilisation du framework **Eureka**, développé par NVIDIA, pour générer automatiquement des fonctions de récompense grâce à un grand modèle de langage (LLM).

### Résultats obtenus
En pratique, **Eureka n’a pas donné de bons résultats** dans ce projet.  
La majorité des itérations ont obtenu **0 % de success rate** lors de l’évaluation, malgré des **récompenses moyennes croissantes** pendant l’entraînement.  
Une évaluation a atteint **10 %**, mais cela n’est **pas statistiquement significatif**, car elle ne portait que sur **10 épisodes**.

> 🏆 Meilleure fonction générée : `training_sessions/eureka_session_20251031_131836/reward_functions/reward_iter2.py`

### Pourquoi Eureka a échoué
L’échec du modèle ne provient pas forcément du framework lui-même, mais plutôt d’un **bug technique** survenu dans la pipeline d’entraînement ou d’évaluation.  
Nous avons observé plusieurs erreurs dans la génération automatique de code, notamment la **continuité du code après un `return`** au lieu de réécrire la fonction complète.  
Il est également possible qu’une **erreur de configuration** soit apparue lors de la fusion entre le projet Drone et le projet Eureka.  
Le **temps d’entraînement très long** n’a pas permis de corriger ces problèmes avant la date de rendu.

### Voies d’amélioration
Pour améliorer les résultats avec Eureka, il serait pertinent de :  
- Corriger la génération de code pour s’assurer que chaque fonction de récompense soit réécrite proprement.  
- Mettre en place une **vérification automatique** du code avant chaque entraînement.  
- Optimiser la **configuration et la durée des itérations** pour accélérer les cycles d’apprentissage et fiabiliser les résultats.

---
---

## Conseils & bonnes pratiques
- **Chemins de modèles** : stockez vos `.zip` d’agents dans `agents/` et mettez à jour `model_paths` / `agents`.
- **`fixed_map` vs `random_start`** : utilisez `fixed_map=True` pour des comparaisons cohérentes, mais gardez `random_start=True` pour tester la robustesse.
- **TensorBoard** : surveillez les métriques d’apprentissage (`tensorboard_logs/`).

---

