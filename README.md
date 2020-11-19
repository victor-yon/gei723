# GEI723 - Neurosciences computationnelles et applications en traitement de l'information

## Installer les dépendances

Python 3.8.5  
pip 20.0.2

```
pip install -r requirements.txt
```

## Problème 1 : Locomotion HexaBob

### Exécuter

```
cd problem-1-locomotion
pyhton3 main.py
```

> Le succès de l'exécution devrait être confirmé par une sortie textuelle.

### Paramètres

Dans le fichier `main.py` plusieurs variables sont facilement modifiables :

* `nb_legs_pair` - Le nombre de paires de pattes à simuler (min: 2 | défaut: 3)
* `sensor_front` - Capteur frontal, fait reculer l'exapode au dessus de 0.5 (min: 0 | max: 1 | défaut: 0)
* `sensor_back` - Capteur arrière, plus la valeur est élevée plus l'exapode avance vite (min: 0 | max: 1 | défaut: 0.2)
* `sensor_left` - Capteur gauche, plus la valeur est élevée plus l'exapode tourne a droite (min: 0 | max: 1 | défaut: 0)
* `sensor_right` - Capteur gauche, plus la valeur est élevée plus l'exapode tourne a gauche (min: 0 | max: 1 | défaut: 0)
* `duration` - Durée de la simulation en ms (min: 1 | défaut: 250)

### Fichiers

* `main.py` : Fichier principal à exécuter.
* `HexaBob.py` : Défintion des objets HexaBob et BobLeg. Une partie des méthodes n'est pas utilisée car nous n'avons pas réussi à les exécuter lors des décharges des neurones moteurs. Mais celles-ci n'impactent pas le fonctionnement gloable du système.
* `bob_nn.py` : Assemblage des différentes parties du réseau de neurones et choix des figures à afficher.
* `cpg_nn.py` : Construction du réseau de neurones pour le générateur de rythme central.
* `direction_nn.py` : Construction du réseau de neurones permettant de tourner à droit et à gauche.
* `ground_contact_nn.py` : Construction du réseau de neurones permettant de baisser et lever la patte. Ce qui permet de gérer la marche arrière et avant.
* `legs_nn.py` : Construction du réseau qui gère les 4 neurones moteurs par patte, pour un même groupe de patte (par exemple les impaires gauches).

## Problème 2 : STDP MNIST

### Exécuter

```
cd problem-2-stdp
pyhton3 main.py
```

### Paramètres

Dans le fichier `main.py` plusieurs variables sont facilement modifiables grace à l'objet "SimulationParameters" dont les différentes paramètres et valeurs par défaut sont accessible dans le fichier `simulation_parameters.py`.

Voici les principaux paramètres :

- `run_name` - Nom de la simulation afin de sauvegarder les résultats dans le dossier `out/`. Ne peut pas être deux fois le même, sauf si son nom est "tmp".
- `nb_train_samples` - Nombre d'image d'entrainement : Nombre d'image que l'on présente au réseau pendant la phase d'entrainement
- `nb_test_samples` - Nombre d'image de test : Nombre d'images que l'on présente au réseau pendant la phase de test.
- `tc_theta` et `theta_plus` - Homéostasie : Nous activons ou désactivons le l'homéostasie pour chaque test.
- `nu_pre` et `nu_post` - Taux d'apprentissage : permet de réguler l'apprentissage en augmentant le poids de façon partiel pendant le processus de STDP. Cela permet d'éviter d'atteindre des valeurs de poids trop élevé trop rapidement.
- `nb_excitator_neurons` et `nb_inhibitor_neurons` - Nombre de neurones excitateurs : nombre de neurones que l'on définit pour la couche excitatrice ET inhibitrice.
- `nb_epoch` - Nombre de fois ou l'on présente le jeu de données permet d'améliorer l'apprentissage sur certains types d'image.
- `normalization` - Normalisation des poids : Normalisation des poids par colonnes avant chaque itération d'entrainement.
- `classification_type` - Type de classification : Deux types de classification, "single" et "group". La classification single, prend le neurone qui a déchargé le plus pendant l'entrainement pour un label donné, et la classification pendant le test sera effectué en fonction des 10 neurones labels sélectionnés. Pour la classification en groupe de neurone, chaque neurone de la couche excitatrice est associé au label pour lequel il a le plus déchargé pendant l'entrainement. Pendant la phase de test, le groupe de neurone correspondant à un label dont la moyenne de décharge est la plus grande donnera le label de l'image testée. 

### Fichiers

- `main.py` : Fichier principal pour lancer une simulation.
- `mnist_stdp.py` : Création du réseau de neurones avec Biran2, boucles d'entrainement et de test. Chargement des données, normalisation et choix des neurones de classification.
- `mnist_stdp_out.py` : Défintion des fonctions permettant d'entregistrer les résultats dans des fichiers.
- `mnist_stdp_plots.py` : Définition des méthodes permettant de créer des figures avec matplotlib et de les enregistrer.
- `simulation_parameters.py` : Définition d'un objet de type `dataclass` permettant de stocker les paramètres d'entrainement et leurs valeurs par défaut.
- `stopwatch.py` : Défintion de fonctions utilitaires permettant de mesurer les temps d'exécution.
- `stdp_shape_freq_study/` : Contient les simulations permettant de mesurer les effets de la forme des équations de STDP ainsi que le changement de fréquence d'entrée.
- `out/` : Dossier de sortie des résultats, doit contenir des fichiers texts et images après chaque simulation.
- `data/` : Dossier contenant les fichiers temporaires pour accélérer les simulations.
- `diehl_cook_classification.py` : Notebook à trous utilisé pour tester les premières implémentations. /!\ N'est pas utilisé pour les simulations finales.

# Auteurs

* Soline Bernard
* Antoine Marion
* Victor Yon

_Université de Sherbrooke - Automne 2020_