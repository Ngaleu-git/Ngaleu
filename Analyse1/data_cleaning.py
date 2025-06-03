# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer # pour imputer les valeurs manquantes  

# %%
# Configuration de l'affichage 
pd.set_option('display.max_columns', None) # pour afficher toutes les colonnes
pd.set_option('display.max_rows', None) # pour afficher toutes les lignes
sns.set_style('whitegrid')  # Utilisation du style whitegrid de seaborn
plt.rcParams['figure.figsize'] = (10, 6)  # Taille par défaut des figures


# %%
print("1. Importation des données...")
# %%
# Chargement des données du Titanic
df = pd.read_csv('data/train.csv')


# %%
#autre script a utiliser
df.dtypes.value_counts() # afficher le noimbre de  type de chaque colonne
df.select_dtypes(['number']) # afficher les colonnes numériques
df.select_dtypes(['object']) # afficher les colonnes catégoriques   
df[df['nom_de_variable'].str.startswith('P')] # afficher dans la partie de nom_de_varaible colonne les modalités qui commencent par P
df[df['nom_de_variable'].str.endswith('e')] # afficher dans la partie de nom_de_varaible colonne les modalités qui finissent par e
df[df['nom_de_variable'].str.contains('e')] # afficher dans la partie de nom_de_varaible colonne les modalités qui contiennent e #autre script a utiliser
df.dtypes.value_counts() # afficher le noimbre de  type de chaque colonne
df.select_dtypes(['number']) # afficher les colonnes numériques
df.select_dtypes(['object']) # afficher les colonnes catégoriques   
df[df['nom_de_variable'].str.startswith('P')] # afficher dans la partie de nom_de_varaible colonne les modalités qui commencent par P
df[df['nom_de_variable'].str.endswith('e')] # afficher dans la partie de nom_de_varaible colonne les modalités qui finissent par e
df[df['nom_de_variable'].str.contains('e')] # afficher dans la partie de nom_de_varaible colonne les modalités qui contiennent e 
df[['nom_de_variable1', 'nom_de_variable2', 'etc....']].apply(lambda x: x.str.strip() if x.dtype == "object" else x) # supprimer les espaces inutiles
df[['nom_de_variable1', 'nom_de_variable2', 'etc....']].apply(np.sum) # somme des colonnes
#Transformer les colonnes 
df['nom_de_variable'] = df['nom_de_variable'].astype(float) # transformer la colonne en float
df[['nom_de_variable1', 'nom_de_variable2', 'etc....']].applymap(float) # transformer les colonnes en float
df['nom_de_variable'] = df['nom_de_variable'].map({'nom_de_modalité1': 1, 'nom_de_modalité2': 2, 'etc....': 3}) # transformer les modalités en nombre



# %%
df.shape
# %%
df.info()   
# %%
# Afficher les premières lignes
print("\nPremières lignes du dataset:")
print(df.head())

# %%
print("\n2. Exploration des données...")
# Informations générales sur le dataset
print("\nInformations sur le dataset:")
print(df.info())

print("\nStatistiques descriptives:")
print(df.describe())

print("\n3. Analyse des valeurs manquantes...")
# Pourcentage de valeurs manquantes par colonne
missing_percent = df.isnull().sum() / len(df) * 100
missing_percent = missing_percent[missing_percent > 0]
print("\nPourcentage de valeurs manquantes par colonne:")
print(missing_percent.sort_values(ascending=False))

# %%
# Visualisation des valeurs manquantes (alternative sans missingno)
plt.figure()
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Distribution des Valeurs Manquantes dans le Dataset Titanic')
plt.savefig('data/missing_values.png')
plt.close()
# %%

print("\n4. Suppression des doublons...")
# Affichage du nombre de doublons
print(f"Nombre de doublons avant suppression: {df.duplicated().sum()}")
# Suppression des doublons
df = df.drop_duplicates()
print(f"Nombre de doublons après suppression: {df.duplicated().sum()}")

print("\n5. Traitement des valeurs aberrantes...")
# Sélection des colonnes numériques pour l'analyse des outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Création d'un boxplot pour chaque variable numérique
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplots des Variables Numériques')
plt.xticks(rotation=45)
plt.savefig('data/outliers_before.png')
plt.close()

# Fonction pour détecter et traiter les outliers
def remove_outliers(df, column): 
    Q1 = df[column].quantile(0.25) # 1er quartile
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Traitement des outliers pour chaque colonne numérique
for col in numeric_cols:
    if col != 'Survived':  # On ne traite pas la variable cible
        df = remove_outliers(df, col)

# Visualisation après traitement des outliers
plt.figure(figsize=(15, 6))
sns.boxplot(data=df[numeric_cols])
plt.title('Boxplots des Variables Numériques après Traitement des Outliers')
plt.xticks(rotation=45)
plt.savefig('data/outliers_after.png')
plt.close()

print("\n6. Nettoyage des données...")
# Suppression des colonnes non pertinentes
df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Imputation des valeurs manquantes
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns # sélectionner les colonnes numériques
categorical_cols = df_clean.select_dtypes(include=['object']).columns # sélectionner les colonnes catégoriques

# Imputation pour les variables numériques (Age)
numeric_imputer = SimpleImputer(strategy='median')
df_clean[numeric_cols] = numeric_imputer.fit_transform(df_clean[numeric_cols]) # imputer les valeurs manquantes avec la médiane

# Imputation pour les variables catégorielles (Embarked)
categorical_imputer = SimpleImputer(strategy='most_frequent')
df_clean[categorical_cols] = categorical_imputer.fit_transform(df_clean[categorical_cols])

# Vérification qu'il n'y a plus de valeurs manquantes
print("\nValeurs manquantes après nettoyage:")
print(df_clean.isnull().sum())

print("\n7. Création des visualisations...")
# Distribution de l'âge
plt.figure()
sns.histplot(data=df_clean, x='Age', kde=True, bins=30)
plt.title('Distribution de l\'âge des passagers')
plt.savefig('data/age_distribution.png')
plt.close()

# Taux de survie par classe
plt.figure()
sns.barplot(data=df_clean, x='Pclass', y='Survived')
plt.title('Taux de survie par classe')
plt.ylabel('Taux de survie')
plt.savefig('data/survival_by_class.png')
plt.close()

# Taux de survie par sexe
plt.figure()
sns.barplot(data=df_clean, x='Sex', y='Survived')
plt.title('Taux de survie par sexe')
plt.ylabel('Taux de survie')
plt.savefig('data/survival_by_sex.png')
plt.close()

# Matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(df_clean[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation')
plt.savefig('data/correlation_matrix.png')
plt.close()

print("\n8. Sauvegarde des données nettoyées...")
# Sauvegarder le dataset nettoyé
df_clean.to_csv('data/titanic_cleaned.csv', index=False)
print("Dataset nettoyé sauvegardé dans 'data/titanic_cleaned.csv'")

# Afficher les premières lignes du dataset nettoyé
print("\nPremières lignes du dataset nettoyé:")
print(df_clean.head())

print("\nTraitement terminé ! Les visualisations ont été sauvegardées dans le dossier 'data/'") 
# %%
