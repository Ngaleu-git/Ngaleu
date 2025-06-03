#%%
####Importation des libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
import os

#%% 
## Importation des données
data = pd.read_csv('personnes.csv', sep=',' ,encoding='utf-8')
data

#Detecter les erreurs
#%%
data.isnull().sum()
#%%
#Fonction pour detecter les valeurs aberrantes
def detect_outliers(data, column): 
    """
    Detect outliers in a given column of a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    column (str): The name of the column to check for outliers.
    
    Returns:
    pd.DataFrame: A DataFrame containing the outliers.
    """
    col = data[column]
    Q1 = data[column].quantile(0.25)  # First quartile (Q1)
    Q3 = data[column].quantile(0.75)  # Third quartile (Q3)
    IQR = Q3 - Q1  # Interquartile range (IQR)
    
    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Return rows where the column value is an outlier
    return data[(col < lower_bound) | (col > upper_bound)]

# %%
detect_outliers(data, 'taille')
# %%
#detection des doublons
data.loc[data['email'].duplicated(keep=False),:]
# %%
# Types de données détectés
data.dtypes
# %%
#DEtection des valeurs numeriques dans les colonnes qualitatives et des erreurs qualitatives
data['pays'].value_counts()

variables_qualitatives = ['pays', 'email','prenom']
#On initialise un DataFrame vide pour accumuler progressivement les lignes avec erreurs détectées dans chaque colonne, sans écraser les résultats à chaque boucle.
resulat = pd.DataFrame()

for i in variables_qualitatives:
    temp = data[data[i].apply(lambda x: str(x).isnumeric())].copy()
    temp["colonne_probleme"] = i
    resulat = pd.concat([resulat, temp])

print(resulat)
# %%
data['email'].value_counts()

# %%
data['email']
data[data['email']]  
# %%%
# On va essayer de detecter les valeurs qui ne sont pas des emails


# %%
#  Formats de date incorrects 5                                                                                                                                                                                                                                                                                                                                
for col in data.columns:
    try: # Tu mets le code qui pourrait provoquer une erreur.
        pd.to_datetime(data[col])
    except (ValueError, TypeError): # Si une erreur se produit, on l'attrape ici.
        print(f"Erreur de format de date dans la colonne : {col}")
        print(f"- {col}")
# %%²²²²²²²²²²²²²²²²²²
# Colonnes avec cellules contenant plusieurs valeurs
for col in data.select_dtypes(include='object'):
    if data[col].str.contains(',|;|/').any():
        print(f"- {col}")
# %%
data[data['taille'] > 30]

####data2#################
# %%
pd.set_option('display.max_rows', data2.shape[0]) 
# %%
####
data2 =pd.read_csv('tableau.csv', sep=';' ,encoding='utf-8')
data2
# %%
data2.shape

# %%
data2.head()
# %%
data2['Dept'].value_counts()
# %%
data2.isnull().sum()
# %%
data2['Temps'] = pd.to_datetime(data2['Temps'], errors='coerce')
# %%
data2
# %%
invalid_values = data2[~data2['Temps'].str.match(r'^\d{2}:\d{2}:\d{2}$')]
print(invalid_values)
# %%
data2.info()


# %%    
data2['Temps_secondes'] =pd.to_numeric(data2["Temps_secondes"], errors='coerce')
# %%
## detection des valeurs aberrantes
valeur_numerique = ['Position','Age','Temps_secondes']
def detect_outliers(data2, column): 
    """
    Detect outliers in a given column of a DataFrame using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    column (str): The name of the column to check for outliers.
    
    Returns:
    pd.DataFrame: A DataFrame containing the outliers.
    """
    col = data2[column]
    Q1 = data2[column].quantile(0.25)  # First quartile (Q1)
    Q3 = data2[column].quantile(0.75)  # Third quartile (Q3)
    IQR = Q3 - Q1  # Interquartile range (IQR)
    
    # Calculate lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Return rows where the column value is an outlier
    return data2[(col < lower_bound) | (col > upper_bound)]

# %%
for i in valeur_numerique:

    print(f"Valeurs aberrantes pour la colonne {i}:")
    outliers = detect_outliers(data2, i)
    if not outliers.empty: # empty est une méthode qui renvoie True si le DataFrame est vide
       # pour vérifier loa condition if l'outlers ne doit pas être vide
        print(outliers)
    else:
        print("Aucune valeur aberrante détectée.")
# %%
data2.loc[data2.duplicated(), :] # pour les doublons

# %%
data2.Sexe.value_counts() # pour les valeurs uniques 

# %%
data2.loc[data2.Temps.isnull(),:] # pour les valeurs manquantes
# %%
data2.iloc[250,:][2]
# %%
data.describe(include=['O']) # pour la description des variables qualitatives
# %%
#analyse d'une variable quantitative en représentant les effectifs et les fréquences et les fréquences cumulées
#Distribution empirique est une représentation de la répartition des valeurs d'une variable dans un échantillon. count_values est une méthode de pandas qui renvoie le nombre d'occurrences de chaque valeur unique dans une série.
effectifs = data["quart_mois"].value_counts()
modalites = effectifs.index # l'index de effectifs contient les modalités

tab = pd.DataFrame(modalites, columns = ["quart_mois"]) # création du tableau à partir des modalités
tab["n"] = effectifs.values
tab["f"] = tab["n"] / len(data) # len(data) renvoie la taille de l'échantillon
tab = tab.sort_values("quart_mois") # tri des valeurs de la variable X (croissant)
tab["F"] = tab["f"].cumsum() # cumsum calcule la somme cumulée

# %%
data['montant'].skew() # qui est une mesure d'asymétrie de la distribution d'une variable aléatoire continue.

"""
Si γ1=0
 alors la distribution est symétrique.

Si γ1>0
 alors la distribution est étalée à droite.

Si γ1<0
 alors la distribution est étalée à gauche.

 L'étude de l'asymétrie d'une distribution, c'est chercher qui de la médiane ou de la moyenne est la plus grande. 
 Une distribution est dite symétrique si elle présente la même forme de part et d’autre du centre de la distribution. Dans ce cas : Mode=Med=x¯¯¯
 
Une distribution est étalée à droite (ou oblique à gauche, ou présentant une asymétrie positive) si : Mode<Med<x¯¯¯
  De même, elle est étalée à gauche (ou oblique à droite) si Mode>Med>x¯¯¯
 
   """
# %%
data['montant'].kurtosis() # qui est une mesure de la "pointedness" ou de l'aplatissement d'une distribution de probabilité.
"""
Le kurtosis empirique n'est pas une mesure d'asymétrie, mais c'est une mesure d'aplatissement. L’aplatissement peut s’interpréter à la condition que la distribution soit symétrique"
Si γ2=0
 , alors la distribution a le même aplatissement que la distribution normale.

Si γ2>0
 , alors elle est moins aplatie que la distribution normale : les observations sont plus concentrées.

Si γ2<0
 , alors les observations sont moins concentrées : la distribution est plus aplatie.

"""


# %%
#En résumé, plus la courbe de Lorenz est proche de la première bissectrice, plus la répartition est égalitaire.
#Nous avons dit que la courbe de Lorenz est un escalier de hauteur 1. Le salaire médial, c'est simplement le salaire de la personne qui se trouve à la moitié de la hauteur : 0,5.
#La courbe de Lorenz est un escalier de hauteur 1. Le salaire médial, c'est simplement le salaire de la personne qui se trouve à la moitié de la hauteur : 0,5.
depenses = data[data['montant'] < 0]
dep = -depenses['montant'].values
n = len(dep)
lorenz = np.cumsum(np.sort(dep)) / dep.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
plt.plot(xaxis,lorenz,drawstyle='steps-post')
plt.show()

# %%
#La courbe de Lorenz n'est pas une statistique, c'est une courbe ! Du coup, on a créé l'indice de Gini, qui résume la courbe de Lorenz.
#Le coefficient de GINI permet d'évaluer de façon chiffrée cette répartition. Il correspond à deux fois l'aire sous la courbe de Lorenz.
AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n # Surface sous la courbe de Lorenz. Le premier segment (lorenz[0]) est à moitié en dessous de 0, on le coupe donc en 2, on fait de même pour le dernier segment lorenz[-1] qui est à moitié au dessus de 1.
S = 0.5 - AUC # surface entre la première bissectrice et le courbe de Lorenz
gini = 2*S
gini
# %%
# La méthode  np.cov  renvoie la matrice de covariance, que vous n'avez pas à connaître à ce niveau.
print(np.cov(depenses["solde_avt_ope"],-depenses["montant"],ddof=0)[1,0])

# %%
"""
Le coefficient de corrélation de Pearson ou coefficient de corrélation linéaire permet de compléter numériquement l'analyse de la corrélation.

Ce dernier n'est pertinent que pour évaluer une relation linéaire. Il prend des valeurs entre -1 et 1, et le signe du coefficient indique le sens de la relation.
"""
# %%
# Code de regression linéaire pour la variable Age et Position si il y a une relation linéaire entre les deux variables
import statsmodels.api as sm
Y = dt['Position']
X = dt[["Age"]].copy()
X['intercept'] = 1
result = sm.OLS(Y, X).fit()

a,b = result.params
print(a, b, result.rsquared)