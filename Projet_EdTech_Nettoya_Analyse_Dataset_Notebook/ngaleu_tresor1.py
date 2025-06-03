# Travail Demande
#Creation d'une fonction récursive insertion_valeur(racine, valeur), qui permet d'insérer une valeur dans un arbre
def insertion_valeur(racine, valeur):
    if racine is None:
        return [valeur, None, None]
    if valeur < racine[0]:
            racine[1] = insertion_valeur(racine[1], valeur)
    elif valeur > racine[0]:
            racine[2] = insertion_valeur(racine[2], valeur)
    return racine

#Creation une fonction (non récursive) insertion_liste(racine, valeurs), qui permet d'insérer une liste de valeurs dans l’arbre
def insertion_liste(racine, valeurs):
    for v in valeurs:
        racine = insertion_valeur(racine, v)
    return racine   

#Creation d'une fonction récursive existe(racine, valeur), qui permet de savoir si une valeur existe dans l'arbre 
def existe(racine, valeur):
    if racine is None:
        return False
    if valeur == racine[0]:
        return True
    elif valeur < racine[0]:
        return existe(racine[1], valeur)
    else:
        return existe(racine[2], valeur)
    
#Creation d'une fonction infixe(racine), qui permet de retourner les valeurs des nœuds dans l’ordre croissant (parcours infixe)
def infixe(racine):
    if racine is None:
        return []
    return infixe(racine[1]) + [racine[0]] + infixe(racine[2])  

#Creation d'une fonction min(racine) et max(racine), qui permet de retourner la valeur minimale et maximale de l’arbre  
def min_val(racine):
    if racine is None:
        return None
    while racine[1] is not None:
        racine = racine[1]
    return racine[0]

def max_val(racine):
    if racine is None:
        return None
    while racine[2] is not None:
        racine = racine[2]
    return racine[0]


# Exemple d'utilisation
racine = None
valeurs = [6, 8, 3, 1, 4, 9, 2, 7, 5]
racine = insertion_liste(racine, valeurs)

print("Infixe :", infixe(racine))
print("Existe 7 :", existe(racine, 7))
print("Min :", min_val(racine))
print("Max :", max_val(racine))