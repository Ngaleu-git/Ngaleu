#Créer un modèle d’apprentissage automatique (machine learning) avec XGBoost
#Préparer automatiquement les données (mise à l’échelle, encodage…)
#Trouver les meilleurs paramètres pour XGBoost grâce à GridSearchCV
#Tester le modèle final sur des données de test
#Afficher les performances (précision, rappel, etc.)

#  Objectif : pipeline qui prépare les données (encodage + mise à l’échelle), entraîne un modèle XGBoost, et optimise les hyperparamètres avec GridSearchCV.
# XGBoost (comme tous les modèles basés sur des arbres) ne comprend que des variables numériques.
# Il faut les transformer avant d’envoyer dans XGBoost.
# exemple complet d’un pipeline avec preprocessing + XGBoost + GridSearchCV, exactement comme tu pourrais l’utiliser dans un vrai projet de data science 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Étape 1 – Charger les données
df = pd.read_csv("donnees_patients.csv")

#Séparer les features (X) et la cible (y)
X = df.drop("survie", axis=1)
y = df["survie"]

# Découper en jeu d'entraînement et de test
# 20% des données pour le test, 80% pour l'entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Étape 2 – Définir les colonnes
# Sélectionner les colonnes numériques et catégoriques
# On sélectionne les colonnes numériques et catégoriques
colonnes_numeriques = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
colonnes_categoriques = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Prétraitement avec ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), colonnes_numeriques),
    ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categoriques)
])

#→ Il applique :

#une mise à l’échelle sur les colonnes numériques (StandardScaler),

#un encodage one-hot sur les colonnes catégoriques (OneHotEncoder).

#Le tout de façon automatique.

# Étape 4 – Pipeline complet avec XGBoost
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])
#Ce pipeline enchaîne :

#Le prétraitement des données

#Le modèle XGBoost

# Étape 5 – Définir une grille d’hyperparamètres pour XGBoost
param_grid = {
    "xgb__n_estimators": [100, 200],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.1, 0.01],
    "xgb__subsample": [0.8, 1.0]
}



# Étape 6 – Optimisation avec GridSearchCV
# GridSearchCV teste plusieurs combinaisons d’hyperparamètres, en 5 plis (cross-validation), et choisit la meilleure.
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)

# Étape 7 – Entraînement
#  Il applique le pipeline + la recherche automatique sur le jeu d’entraînement.
grid_search.fit(X_train, y_train)

# Étape 8 – Évaluation
#Il teste le meilleur modèle trouvé sur les données de test.
#Et il affiche : précision, rappel, F1-score, etc.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Meilleurs hyperparamètres :", grid_search.best_params_)
print(classification_report(y_test, y_pred))
