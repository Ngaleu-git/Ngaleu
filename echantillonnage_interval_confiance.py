# 📌 Importation des bibliothèques nécessaires
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# 📌 1. Génération de données suivant une loi normale N(12, 4)
np.random.seed(123)  # Pour la reproductibilité
X = np.random.normal(loc=12, scale=2, size=25)  # 𝜎 = sqrt(4) = 2

# Affichage des premières valeurs
print("Premières valeurs générées :", np.round(X[:5], 3))

# 📌 2. Estimation des paramètres (moyenne et variance)
m_estimee = round(np.mean(X), 3)
variance_estimee = round(np.var(X, ddof=1), 3)  # ddof=1 pour variance d’échantillon

print(f"Moyenne estimée : {m_estimee}")
print(f"Variance estimée : {variance_estimee}")

# 📌 3. Intervalle de confiance à 95%
ic_95 = stats.t.interval(confidence=0.95, df=len(X)-1, loc=np.mean(X), scale=stats.sem(X))
ic_95 = [round(val, 3) for val in ic_95]

print(f"Intervalle de confiance à 95% pour la moyenne : {ic_95}")

# 📌 4. Intervalles de confiance à 90% et 99%
ic_90 = stats.t.interval(0.90, df=len(X)-1, loc=np.mean(X), scale=stats.sem(X))
ic_99 = stats.t.interval(0.99, df=len(X)-1, loc=np.mean(X), scale=stats.sem(X))

print(f"IC à 90% : {[round(val, 3) for val in ic_90]}")
print(f"IC à 99% : {[round(val, 3) for val in ic_99]}")

# 📌 5. Influence de la taille de l’échantillon
X15 = np.random.normal(12, 2, 15)
X100 = np.random.normal(12, 2, 100)
X1000 = np.random.normal(12, 2, 1000)

for sample, name in zip([X15, X100, X1000], ["n=15", "n=100", "n=1000"]):
    ic = stats.t.interval(0.95, df=len(sample)-1, loc=np.mean(sample), scale=stats.sem(sample))
    print(f"{name} → IC à 95% : {[round(val, 3) for val in ic]}")

# 📌 6. Visualisation (optionnelle)
plt.figure(figsize=(10, 5)) #  Crée une nouvelle figure pour le graphique avec une taille de 10 pouces par 5.
plt.hist(X, bins=10, density=True, alpha=0.6, color='skyblue', label='Données simulées') #  Crée une nouvelle figure pour le graphique avec une taille de 10 pouces par 5.
xmin, xmax = plt.xlim() #  Récupère les bornes de l’axe X (gauche et droite) de l’histogramme pour savoir où tracer la courbe.
x = np.linspace(xmin, xmax, 100) # Crée 100 points entre xmin et xmax, pour pouvoir tracer une courbe lisse ensuite.
p = stats.norm.pdf(x, np.mean(X), np.std(X, ddof=1)) # Calcule les valeurs de la densité de la loi normale (PDF) :
"""
centrée sur la moyenne de X,

avec l’écart-type de X corrigé (avec ddof=1).

"""
plt.plot(x, p, 'k', linewidth=2, label='Courbe normale estimée')
"""
➡️ Trace la courbe de Gauss estimée :

x = valeurs de l’axe horizontal,

p = valeurs de la densité normale,

'k' = couleur noire (k = black),

linewidth=2 = épaisseur de la ligne,

label='Courbe normale estimée' = pour la légende.

"""
plt.title('Histogramme des données et ajustement d\'une loi normale')
plt.xlabel('Valeurs')
plt.ylabel('Densité')
plt.legend()
plt.grid(True)
plt.show()

########################################################################################################################################################
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Paramètres
N = 50        # Nombre de répétitions
m = 12       # Vraie moyenne
sigma = 2     # Écart-type
n = 30        # Taille de l'échantillon
alpha = 0.05  # Risque (donc 95% de confiance)

# Fonction pour construire l'intervalle de confiance
def conf_interval(data, alpha=0.05):
    mean = np.mean(data)
    sem = stats.sem(data)  # Erreur standard
    margin = stats.t.ppf(1 - alpha/2, df=len(data) - 1) * sem
    return mean - margin, mean, mean + margin

# Simulation
intervals = []
for _ in range(N):
    sample = np.random.normal(mu, sigma, n)
    intervals.append(conf_interval(sample, alpha))

# Analyse : combien ne couvrent pas la vraie moyenne ?
non_couv = [i for i, (low, _, high) in enumerate(intervals) if not (low <= mu <= high)]
pourcentage_non_couv = len(non_couv) * 100 / N
print(f"{len(non_couv)} intervalles ne contiennent pas la vraie moyenne.")
print(f"Pourcentage : {pourcentage_non_couv:.1f}%")

# Visualisation
for i, (low, mean, high) in enumerate(intervals):
    color = 'red' if i in non_couv else 'green'
    plt.plot([i, i], [low, high], color=color)
    plt.plot(i, mean, 'o', color=color)

plt.axhline(mu, color='blue', linestyle='--', label='Vraie moyenne')
plt.title("Intervalles de confiance pour 50 échantillons")
plt.xlabel("Échantillon")
plt.ylabel("Valeur")
plt.legend()
plt.show()
