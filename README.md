### **Analyse et Prédiction du Risque de Crédit**



Dans le secteur bancaire et financier, l'octroi d'un prêt est une décision critique qui repose sur l'évaluation de la solvabilité d'un client. Une mauvaise estimation peut entraîner des pertes financières significatives pour l'institution si l'emprunteur se retrouve en situation de défaut de paiement.



Aujourd'hui, le volume de demandes de crédit est trop élevé pour une analyse purement manuelle. Le défi est de construire un système capable de :

* Identifier automatiquement les profils à risque.
* Minimiser les erreurs de classification (notamment les "faux négatifs" : prêter à quelqu'un qui ne peut pas rembourser).
* Aider les analystes de crédit à prendre des décisions basées sur des données objectives.



Ce projet implémente un pipeline complet de Data Science pour prédire la probabilité de défaut de crédit. J'ai utilisé **un modèle de Régression Logistique**, reconnu pour sa robustesse et sa facilité d'interprétation dans le milieu bancaire.



Le projet se décompose en plusieurs étapes clés :



1. Exploration et Diagnostic : Analyse de la structure des données et vérification de la qualité (valeurs manquantes, types de variables).
   
2. Nettoyage (Data Wrangling) : Préparation des jeux de données (credit\_risk\_dataset.csv) pour l'entraînement.
   
3. Visualisation : Génération de graphiques pour comprendre les facteurs influençant le risque, comme le revenu ou l'historique de crédit (ex: correlation\_matrix.png, revenu\_vs\_statut.png).
   
4. Modélisation : Application du modèle de classification binaire via la bibliothèque Scikit-learn.



##### **Tech Stack**



	**-Language :** Python

	**-Data Manipulation :** Pandas , Numpy 

	**-Visualisation :** Matplotlib , Seaborn

	**-Machine Learning :** Scikit-learn

	**-Environnement :** VS Code

