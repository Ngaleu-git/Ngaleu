"""
MÉTHODOLOGIE D'ANALYSE DE DONNÉES
================================

Ce fichier contient une méthodologie complète pour analyser des données de manière systématique.
Il inclut des fonctions, des check-lists et des guides d'interprétation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration de base
plt.style.use('seaborn')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def checklist_analyse():
    """
    Check-list complète pour l'analyse de données
    """
    print("""
    CHECK-LIST D'ANALYSE DE DONNÉES
    
    1. PREMIÈRE APPROCHE
    - Charger les données
    - Vérifier la structure (shape, info)
    - Identifier les types de données
    - Détecter les valeurs manquantes
    
    2. ANALYSE DESCRIPTIVE
    - Statistiques de base (describe)
    - Distributions des variables
    - Corrélations entre variables
    
    3. ANALYSE TEMPORELLE (si applicable)
    - Évolution dans le temps
    - Comparaison périodes
    - Tendance générale
    
    4. ANALYSE COMPARATIVE
    - Comparaison entre catégories
    - Performance relative
    - Classement des éléments
    
    5. IDENTIFICATION DES POINTS CLÉS
    - Valeurs extrêmes
    - Variations significatives
    - Écarts importants
    
    6. SYNTHÈSE ET RECOMMANDATIONS
    - Points forts
    - Points d'attention
    - Actions à mener
    """)

def guide_interpretation():
    """
    Guide pour interpréter les résultats statistiques
    """
    print("""
    GUIDE D'INTERPRÉTATION DES RÉSULTATS
    
    1. VARIATIONS
    - > 20% : Variation significative
    - 10-20% : Variation notable
    - < 10% : Variation normale
    
    2. CORRÉLATIONS
    - > 0.7 : Forte corrélation
    - 0.3-0.7 : Corrélation moyenne
    - < 0.3 : Faible corrélation
    
    3. DISTRIBUTIONS
    - Symétrique : Distribution normale
    - Asymétrique : Distribution particulière
    - Bimodale : Deux groupes distincts
    
    4. VALEURS MANQUANTES
    - < 5% : Négligeable
    - 5-20% : À traiter
    - > 20% : Problématique
    """)

def questions_analyse():
    """
    Questions clés à se poser pendant l'analyse
    """
    print("""
    QUESTIONS CLÉS POUR L'ANALYSE
    
    1. SUR LES DONNÉES
    - Quelle est la source des données ?
    - Sont-elles complètes et fiables ?
    - Y a-t-il des valeurs aberrantes ?
    
    2. SUR LES VARIATIONS
    - Les variations sont-elles significatives ?
    - Sont-elles cohérentes avec le contexte ?
    - Y a-t-il des explications évidentes ?
    
    3. SUR LES TENDANCES
    - La tendance est-elle claire ?
    - Est-elle durable ou ponctuelle ?
    - Y a-t-il des saisonnalités ?
    
    4. SUR LES COMPARAISONS
    - Les comparaisons sont-elles pertinentes ?
    - Les écarts sont-ils expliqués ?
    - Y a-t-il des similarités intéressantes ?
    """)

def template_analyse(df):
    """
    Template de base pour l'analyse de données
    """
    # 1. Analyse initiale
    print("\n1. ANALYSE INITIALE")
    print(f"Nombre de lignes : {df.shape[0]}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print("\nTypes de données :")
    print(df.dtypes)
    
    # 2. Statistiques descriptives
    print("\n2. STATISTIQUES DESCRIPTIVES")
    print(df.describe())
    
    # 3. Visualisations de base
    print("\n3. VISUALISATIONS")
    # Ajouter vos visualisations ici
    
    # 4. Analyse détaillée
    print("\n4. ANALYSE DÉTAILLÉE")
    # Ajouter vos analyses spécifiques ici
    
    # 5. Synthèse
    print("\n5. SYNTHÈSE")
    # Ajouter vos conclusions ici

def analyser_donnees_commerciales(df):
    """
    Exemple d'analyse de données commerciales
    """
    # 1. Préparation
    print("\n=== ANALYSE COMMERCIALE ===")
    
    # 2. Calcul des indicateurs
    df['Variation'] = ((df['REEL 2015'] - df['REEL 2014']) / df['REEL 2014'] * 100)
    df['Ecart_Budget'] = df['REEL 2015'] - df['BUDGET 2015']
    
    # 3. Analyse par zone
    for zone in df['espace de vente']:
        zone_data = df[df['espace de vente'] == zone].iloc[0]
        
        print(f"\n--- Analyse de la zone {zone} ---")
        print(f"Variation 2014-2015 : {zone_data['Variation']:.2f}%")
        print(f"Ecart budget : {zone_data['Ecart_Budget']:,.2f}€")
        
        # Interprétation
        if zone_data['Variation'] > 20:
            print("→ Forte croissance : Performance exceptionnelle")
        elif zone_data['Variation'] < -20:
            print("→ Forte baisse : Situation préoccupante")
        else:
            print("→ Évolution normale")

def journal_analyse():
    """
    Template de journal d'analyse
    """
    print("""
    JOURNAL D'ANALYSE
    
    Date : ______________
    Jeu de données : ______________
    
    1. OBSERVATIONS CLÉS
    - Point 1 : ______________
    - Point 2 : ______________
    - Point 3 : ______________
    
    2. QUESTIONS SOULEVÉES
    - Question 1 : ______________
    - Question 2 : ______________
    - Question 3 : ______________
    
    3. HYPOTHÈSES
    - Hypothèse 1 : ______________
    - Hypothèse 2 : ______________
    - Hypothèse 3 : ______________
    
    4. CONCLUSIONS
    - Conclusion 1 : ______________
    - Conclusion 2 : ______________
    - Conclusion 3 : ______________
    """)

def visualisations_base(df):
    """
    Création de visualisations de base
    """
    # 1. Histogrammes pour les variables numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution de {col}')
        plt.savefig(f'distribution_{col}.png')
        plt.close()
    
    # 2. Diagrammes en barres pour les variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution de {col}')
        plt.xticks(rotation=45)
        plt.savefig(f'barplot_{col}.png')
        plt.close()
    
    # 3. Matrice de corrélation
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Matrice de Corrélation')
        plt.savefig('correlation_matrix.png')
        plt.close()

def detecter_anomalies(df):
    """
    Détection des valeurs aberrantes
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    anomalies = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies[col] = {
            'lower': df[df[col] < lower_bound].shape[0],
            'upper': df[df[col] > upper_bound].shape[0]
        }
    
    return anomalies

def analyser_tendances(df, date_col=None):
    """
    Analyse des tendances temporelles
    """
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Analyse de la tendance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            df[col].plot()
            plt.title(f'Tendance de {col}')
            plt.savefig(f'tendance_{col}.png')
            plt.close()

def generer_rapport(df):
    """
    Génération d'un rapport d'analyse complet
    """
    print("\n=== RAPPORT D'ANALYSE ===")
    
    # 1. Informations générales
    print("\n1. INFORMATIONS GÉNÉRALES")
    print(f"Nombre d'observations : {df.shape[0]}")
    print(f"Nombre de variables : {df.shape[1]}")
    
    # 2. Qualité des données
    print("\n2. QUALITÉ DES DONNÉES")
    missing_values = df.isnull().sum()
    print("\nValeurs manquantes par colonne :")
    print(missing_values[missing_values > 0])
    
    # 3. Statistiques descriptives
    print("\n3. STATISTIQUES DESCRIPTIVES")
    print(df.describe())
    
    # 4. Détection des anomalies
    print("\n4. DÉTECTION DES ANOMALIES")
    anomalies = detecter_anomalies(df)
    for col, values in anomalies.items():
        print(f"\n{col}:")
        print(f"- Valeurs inférieures anormales : {values['lower']}")
        print(f"- Valeurs supérieures anormales : {values['upper']}")
    
    # 5. Visualisations
    print("\n5. VISUALISATIONS")
    visualisations_base(df)
    
    print("\nRapport généré avec succès !")

def creer_graphiques_professionnels(df):
    """
    Crée des graphiques professionnels avec un style moderne et des annotations
    """
    # Configuration du style
    plt.style.use('seaborn')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # 1. Graphique de comparaison des performances
    plt.figure(figsize=(14, 8))
    x = range(len(df['espace de vente']))
    width = 0.35
    
    bars_2014 = plt.bar([i - width/2 for i in x], df['REEL 2014'], width, 
                        label='2014', color='#3498db', alpha=0.8)
    bars_2015 = plt.bar([i + width/2 for i in x], df['REEL 2015'], width, 
                        label='2015', color='#2ecc71', alpha=0.8)
    
    # Ajout des valeurs sur les barres
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}€',
                    ha='center', va='bottom', fontsize=10)
    
    add_value_labels(bars_2014)
    add_value_labels(bars_2015)
    
    plt.xlabel('Zones géographiques', fontweight='bold')
    plt.ylabel('Chiffre d\'affaires (€)', fontweight='bold')
    plt.title('Comparaison des performances 2014 vs 2015 par zone', 
              fontweight='bold', pad=20)
    plt.xticks(x, df['espace de vente'], rotation=45, ha='right')
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Graphique des écarts budget vs réel
    plt.figure(figsize=(14, 8))
    df['Ecart Budget-Réel 2015'] = df['REEL 2015'] - df['BUDGET 2015']
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df['Ecart Budget-Réel 2015']]
    bars = plt.bar(df['espace de vente'], df['Ecart Budget-Réel 2015'], 
                   color=colors, alpha=0.8)
    
    # Ajout des valeurs et annotations
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -10
        else:
            va = 'bottom'
            offset = 10
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:,.0f}€',
                ha='center', va=va, fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Zones géographiques', fontweight='bold')
    plt.ylabel('Ecart Budget-Réel (€)', fontweight='bold')
    plt.title('Ecart entre le budget et le réel 2015 par zone', 
              fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('budget_vs_reel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Graphique d'évolution des performances
    plt.figure(figsize=(14, 8))
    for zone in df['espace de vente']:
        zone_data = df[df['espace de vente'] == zone].iloc[0]
        plt.plot(['2014', '2015'], 
                 [zone_data['REEL 2014'], zone_data['REEL 2015']], 
                 marker='o', markersize=8, linewidth=2, label=zone)
    
    plt.xlabel('Année', fontweight='bold')
    plt.ylabel('Chiffre d\'affaires (€)', fontweight='bold')
    plt.title('Evolution des performances 2014-2015 par zone', 
              fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evolution_performances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Graphique de répartition du CA 2015
    plt.figure(figsize=(14, 8))
    total_2015 = df['REEL 2015'].sum()
    parts = df['REEL 2015'] / total_2015 * 100
    
    # Utilisation d'une palette de couleurs modernes
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(parts)))
    
    # Création du camembert avec des effets d'ombre
    plt.pie(parts, labels=df['espace de vente'], 
            autopct='%1.1f%%', startangle=90,
            colors=colors, shadow=True,
            textprops={'fontsize': 12})
    
    plt.title('Répartition du chiffre d\'affaires par zone en 2015', 
              fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('repartition_ca_2015.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Graphique de variation en pourcentage
    plt.figure(figsize=(14, 8))
    df['Variation'] = ((df['REEL 2015'] - df['REEL 2014']) / df['REEL 2014'] * 100)
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in df['Variation']]
    bars = plt.bar(df['espace de vente'], df['Variation'], 
                   color=colors, alpha=0.8)
    
    # Ajout des valeurs et annotations
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            offset = -5
        else:
            va = 'bottom'
            offset = 5
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:.1f}%',
                ha='center', va=va, fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Zones géographiques', fontweight='bold')
    plt.ylabel('Variation en %', fontweight='bold')
    plt.title('Variation des performances 2014-2015 par zone', 
              fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('variation_percentages.png', dpi=300, bbox_inches='tight')
    plt.close()

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les données
    try:
        df = pd.read_csv('donnee_commercial.csv')
        
        # Afficher la check-list
        checklist_analyse()
        
        # Générer le rapport
        generer_rapport(df)
        
        # Créer les graphiques professionnels
        creer_graphiques_professionnels(df)
        
        # Analyser les données commerciales
        analyser_donnees_commerciales(df)
        
    except Exception as e:
        print(f"Erreur lors de l'analyse : {str(e)}") 