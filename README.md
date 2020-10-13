# GEI723 - Neurosciences computationnelles et applications en traitement de l'information

## Installer les dépendances

Python 3.8.5  
pip 20.0.2

> pip install -r requirements.txt

## Problème 1 : Locomotion HexaBob

### Exécuter

> cd problem-1-locomotion  
> pyhton3 main.py

### Paramètres

Dans le fichier `main.py` plusieurs variables sont facilement modifiables :

* `nb_legs_pair` - Le nombre de pair de pattes à simuler (min : 2 | défaut : 3)
* `sensor_front` - Capteur frontal, fait reculer l'exapode au dessus de 0.5 (min : 0 | max : 1 | défaut : 0)
* `sensor_back` - Capteur arrière, plus la valeur est élevée plus l'exapode avance vite (min : 0 | max : 1 | défaut : 0.2)
* `sensor_left` - Capteur gauche, plus la valeur est élevée plus l'exapode tourne a droite (min : 0 | max : 1 | défaut : 0)
* `sensor_right` - Capteur gauche, plus la valeur est élevée plus l'exapode tourne a gauche (min : 0 | max : 1 | défaut : 0)
* `duration` - Durée de la simulation en ms (min : 1 | défaut : 300)

### Fichiers

* `main.py` : Fichier principal à exécuter.
* `HexaBob.py` : Défintion des objets HexaBob et BobLeg. Une partie des méthodes ne sont pas utilisées car nous n'avons pas réussi à les exécuter lors des décharges des neurones moteurs.
* `bob_nn.py` : Assemblage des différentes partie du réseau de neurones et choix des figures à afficher.
* `cpg_nn.py` : Construction du réseau de neurones pour le générateur de rythme central.
* `direction_nn.py` : Construction du réseau de neurones qui permet de tourner à droit et à gauche.
* `ground_contact_nn.py` : Construction du réseau de neurones qui permet de baisser et lever la patte. Ce qui permet de gérer la marche arrière et avant.
* `legs_nn.py` : Construction du réseau qui gère les 4 neurones moteurs par patte pour un même groupe de patte (par exemple les impaires gauche).

# Auteurs

* Soline Bernard
* Antoine Marion
* Victor Yon