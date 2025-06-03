# %%
### importation des bibliothèques pourt le traitement de données
import pandas as pd
import numpy as np          
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.seasonal import seasonal_decompose # pour ananlyser les séries temporelles
from statsmodels.tsa.stattools import adfuller # pour tester la stationnarité de la série temporelle    


# %%
pd.set_option('display.max_columns', None) # pour afficher toutes les colonnes
pd.set_option('display.max_rows', None) # pour afficher toutes les lignes
sns.set_style('whitegrid')  # Utilisation du style whitegrid de seaborn
plt.rcParams['figure.figsize'] = (10, 6)  # Taille par défaut des figures



# %%
##Imporation de la donnée
## parse_dates=['Order Date', 'Ship Date'] : pour la conversion automatique des dates
df = pd.read_csv('train.csv', sep=',', header=0,encoding='UTF-8',parse_dates=['Order Date', 'Ship Date'], dayfirst=True)
df.head()
# %%
# mesure de la taille du dataframe
df.shape

# %%
pd.set_option('display.max_columns', df.shape[1]) # pour afficher toutes les colonnes
pd.set_option('display.max_rows', df.shape[0]) # pour afficher toutes les lignes
# %%
df.info()

# %%
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format="%d/%m/%Y")
df['Order Date'] = pd.to_datetime(df['Order Date'], format="%d/%m/%Y")


# %%
# Vérification des types de données après conversion
df.dtypes

# %%
# Analyse des valeursc manquantes
df.isnull().sum()
# %%
# Suppression des colonnes avec plus de 50% de valeurs manquantes et des colonnes non pertinentes
df.drop(['Row ID', 'Order ID', 'Customer ID', 'Product ID'], axis=1, inplace=True)
# %%
df.head()
# %%
df.columns

# %%
df.isnull().sum()

# %%
# remplacement des valeurs manquantes par la valeur 
df.describe()

# %%
# Convertie la colonne 'Postal Code' en chaîne de caractères
df['Postal Code'] = df['Postal Code'].astype(str)

#%%
df.dtypes
# %%
df['Postal Code'].value_counts().max()
#%%
df['Postal Code'].value_counts()[df['Postal Code'].value_counts() == df['Postal Code'].value_counts().max()]



# %%
# Remplacement des valeurs manquantes par la valeur 0
df["Postal Code"] = df["Postal Code"].fillna(10035)
# %%
df.isnull().sum()

# %%
df.info()


# %%
df['Order Date'].value_counts().reset_index()

## Visualisation de la distribution des données

## on arrive a extrait le mois sous forme de chaîne de caractères car la variable order date est sous forme de index
# %%
# On va grouper les données par date de commande et calculer la somme des ventes pour chaque date et mettre les index
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales
# %%
# trier par ordre croissant
daily_sales = daily_sales.sort_values('Order Date')
daily_sales

# %%
# Visualisation de la série temporelle des ventes quotidiennes en mettant order date en index
# resample('D') : pour regrouper les données par jour
#resample('ME') : pour regrouper les données par mois
# resample() est une fonction de Pandas qui permet de regrouper des données temporelles (comme des dates) par période (jour, mois, semaine…) pour ensuite appliquer une fonction d’agrégation (comme .sum(), .mean(), etc.).
#.resample('ME').mean()	Moyenne journalière de chaque mois (activité type)
#.resample('ME').sum()	Total des ventes du mois (volume global)
#❌ Sans .mean() ou .sum()	Erreur, car Pandas ne sait pas quoi faire des groupes
monthly_sales = daily_sales.set_index('Order Date').resample('ME').sum()
monthly_sales.index
# %%
import matplotlib.dates as mdates

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['Sales'], marker='o')

# Ajoute un format de date pour afficher tous les mois
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # chaque mois
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format : année-mois
plt.xticks(rotation=45)  # pour que ça ne se chevauche pas

plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()
# %%
df2 = monthly_sales.copy()
df2['year'] = df2.index.year # pour extraire l'année
df2
# %%
# strftime('%b') : pour extraire le mois sous forme de chaîne de caractères (ex: 'Jan', 'Feb', 'Mar', etc.)
# strftime('%m') : pour extraire le mois sous forme de nombre (ex: '01', '02', '03', etc.)
df2['month'] = df2.index.strftime('%b')
df2
# %%
# on arrive a extrait le mois sous forme de chaîne de caractères car la variable order date est sous forme de index
df2['month_num'] = df2.index.month # pour extraire le mois sous forme de nombre (ex: '01', '02', '03', etc.)
df2
# %%
