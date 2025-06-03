Objectif de l'analyse est de comment anticiper les baisses de ventes saisonnières d'une entreprise afin d'optimiser les stocks et éviter les pertes financières.
 Pour cela nous allons:
 
## Étapes du Projet
1. Importation et exploration des données
2. Analyse des valeurs manquantes
3. Nettoyage des données
4. Visualisation des données
5. Analyse statistique

## Description de la donnée
Anglais	Français
Order Date	:Date de commande
Ship Date	:Date d'expédition
Ship Mode	:Mode d'expédition
Customer Name	:Nom du client
Segment	:Segment (ou Type de clientèle)
Country	:Pays
City	:Ville
State	:État (ou Région administrative)
Postal :Code postal
Region	:Région
Category	:Catégorie
Sub-Category	:Sous-catégorie
Product Name	!Nom du produit
Sales	:Ventes



## Idées de questions d'analyse sur cette base :
Performance commerciale :

Quel est le chiffre d'affaires total réalisé par région ?

Quels sont les 5 produits les plus vendus en termes de quantité ?

Quels produits génèrent le plus de pertes ?

Analyse client :

Quels clients ont passé le plus de commandes ?

Quelle est la valeur moyenne d'une commande par client ?

Existe-t-il des segments de clients particulièrement rentables ?

Analyse temporelle :

Comment les ventes évoluent-elles mois par mois ?

Quel trimestre est le plus rentable ?

Y a-t-il une saisonnalité des ventes observable ?

Analyse géographique :

Quels États ou Villes génèrent le plus de profit ?

Y a-t-il des régions où la rentabilité est faible malgré un volume de ventes élevé ?

Analyse logistique :

Quel mode de livraison est le plus utilisé ?

Quel est l'impact du mode d'expédition sur le délai de livraison ou la satisfaction ?

Analyse des remises :

Les remises augmentent-elles réellement les ventes ?

Quel est l'impact des remises sur les marges bénéficiaires ?

Utiliser seasonal_decompose pour décomposer les ventes en :

Tendance (est-ce que les ventes montent ou descendent sur l'année ?)

Saisonnalité (y a-t-il des baisses régulières chaque été, chaque hiver ?)

Résidus (les anomalies)

Utiliser adfuller pour tester la stationnarité :

Si les ventes sont stationnaires, tu peux utiliser des modèles simples pour prévoir.

Si elles ne le sont pas (ex: une tendance forte), il faudra transformer la série (différenciation, etc.) avant de prévoir.

Proposer des solutions basées sur les résultats :

Ajuster les stocks en fonction des prévisions de baisse.

Lancer des campagnes marketing avant les baisses de ventes pour limiter la perte.

Planifier les achats et la production selon les cycles identifiés.