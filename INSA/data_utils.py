# -*- coding: utf-8 -*-
"""
fonctions accompagnant app_annotation
"""

import pandas as pd

def date_francaise(date):
    """
    Convertit une date en format lisible selon les conventions françaises.
    
    Parameters
    ----------
    date : str
        Date au format "AAAA-MM-JJ" 
    
    Returns
    -------
    str
        Date formatée sous forme de chaîne de caractères.
    """
    date_anglais = pd.to_datetime(date,format='%Y-%m-%d').strftime('%d %B %Y')
    
    correspondance = {"January" : "Janvier", "February" : "Février", "March" : "Mars",
                "April" : "Avril", "May":"Mai", "June":"Juin", "July":"Juillet",
                "August":"Août", "September":"Septembre", "October":"Octobre",
                "November":"Novembre", "December":"Décembre"}
    
    for mois_anglais in correspondance.keys():
        if mois_anglais in date_anglais:
            mois_francais = correspondance[mois_anglais]
            return date_anglais.replace(mois_anglais, mois_francais)

def format_species(espece):
    """
    Normalise le format des noms d’espèces.
    Retire les suffixes des noms d'espèces (année de découverte et découvreur)
    Format obtenu : "Genre espèce" ou "Genre espèce subsp. sous-espèce" ou "croisée1 x croisée2"
    Cette fonction assure une cohérence d’écriture pour les comparaisons
    et les jointures entre différentes sources de données.
    
    Parameters
    ----------
    espece : str
        nom d'une espèce
        
    Returns
    -------
    formatted_esp : str
        nom de l'espèce au bon format
        
    """
    l = espece.split()
    if "subsp." in l:
        return " ".join(l[:4])
    elif "x" in l:
        return " ".join(l[:3])
    else:
        return " ".join(l[:2])