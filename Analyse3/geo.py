import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


# Lis le fichier shapefile téléchargé
gdf = gpd.read_file("chemin/vers/le/dossier/ne_110m_admin_0_countries.shp")
gdf.plot()
france = gdf[gdf.name == "France"]

# Simuler des données régionales
regions = ['Île-de-France', 'Nouvelle-Aquitaine', 'Auvergne-Rhône-Alpes', 'Occitanie', 'Bretagne']
valeurs = [85, 45, 60, 30, 70]  # Ex: taux de chômage ou autre

# Créer un DataFrame avec les données
data = pd.DataFrame({
    'region': regions,
    'valeur': valeurs
})

# Tu auras besoin d'un shapefile des régions françaises pour bien faire la carte
# Ici on suppose que tu as un GeoDataFrame `regions_gdf` avec une colonne 'region'
# et que tu fusionnes tes données avec ce shapefile :
# merged = regions_gdf.merge(data, on="region")

# Pour l'exemple, on fait une carte du pays entier
france.plot()
plt.title("Carte de la France (exemple de base)")
plt.show()
