# Prédiction de prix Airbnb — régression à vecteurs de support

Une application Streamlit qui estime le prix d'une annonce Airbnb à partir de ses caractéristiques. Le modèle est une **régression à vecteurs de support à noyau RBF**, précédée d'une normalisation et d'une réduction de dimension par ACP.

## La chaîne

Les trois objets sont sérialisés et chargés au démarrage, dans cet ordre :

| Étape | Objet | Rôle |
| --- | --- | --- |
| **1. Normalisation** | `models/scaler.pkl` | Met toutes les variables à la même échelle. Indispensable : un noyau RBF mesure des distances, et une variable en milliers d'unités écraserait toutes les autres. |
| **2. Réduction** | `models/pca_model.pkl` | Projette sur les composantes principales. Réduit le bruit et accélère la prédiction. |
| **3. Prédiction** | `models/model_rbf.pkl` | La SVR à noyau gaussien produit l'estimation de prix. |

L'ordre compte, et une erreur courante consiste à l'inverser : l'ACP doit être ajustée **après** la normalisation, sinon elle privilégie les variables de grande amplitude plutôt que celles qui portent l'information.

## Pourquoi un noyau RBF

Le prix d'un logement ne varie pas linéairement avec ses caractéristiques. Une chambre supplémentaire ne vaut pas le même supplément selon le quartier, et l'effet de la capacité d'accueil sature. Le noyau gaussien permet à la SVR de capturer ces effets sans qu'on ait à les décrire à la main.

Contrepartie : le modèle devient une boîte noire. Il donne un prix, pas une explication du prix.

## Contenu du dépôt

| Fichier | Rôle |
| --- | --- |
| `main.py` | L'application Streamlit : saisie des caractéristiques et prédiction |
| `test.py` | Vérifications sur le chargement et la prédiction |
| `models/` | Modèle, normalisateur et ACP, sérialisés |
| `requirements.txt` | streamlit, pandas, numpy, scikit-learn, joblib |

## Mise en route

```bash
pip install -r requirements.txt
streamlit run main.py
```

Les trois modèles étant fournis, rien n'est à réentraîner.

## Limites

Le dépôt contient les modèles mais **pas le carnet d'entraînement** : on ne peut ni vérifier les performances annoncées, ni savoir sur quelle ville ni sur quelle période les données ont été collectées. Un marché Airbnb est très local et bouge vite ; un modèle entraîné sur une ville et une saison données ne se transporte pas.

Ajouter le carnet d'entraînement, avec les métriques sur le jeu de test, rendrait le tout vérifiable.
