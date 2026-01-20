# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

"""
Module pour le calcul des métriques
"""

def indice_jaccard(set1, set2):
    """
    Calcule l'indice de Jaccard entre 2 sets qui contiennent des noms d'espèces.
    L'indice de Jaccard représente la similarité entre 2 ensembles, et est utilisé pour comparer 
    les relevés à des listes de références des milieux isérois.

    Parameters
    ----------
    set1 : set
        set contenant une liste d'espèces (observées ou caractéristiques)
        
    set2 : set
        set contenant une liste d'espèces (observées ou caractéristiques)
        
    Returns
    -------
    cardinal_inter : int
        cardinal de l'intersection des 2 sets, nombre d'espèces présentes dans les 2 sets
    indice : float
        indice de Jaccard
    """
    cardinal_union = len(set1|set2)
    cardinal_inter = len(set1 & set2)
    indice = cardinal_inter/cardinal_union
    return cardinal_inter, indice

def compute_mean_atypicity_per_releve(data, atypicity_column):
    """
    Calcule la moyenne de l'atypicité par relevé dans tout le dataframe.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame contenant au moins les colonnes 'Code_Releve' et la colonne d'atypicité spécifiée
    
    Returns
    -------
    pd.Series
        Moyenne de l'atypicité pour chaque 'Code_Releve'
    """
    return data[["Code_Releve", atypicity_column]].groupby(["Code_Releve"]).mean()[atypicity_column].dropna()


def compute_proportion_lower_atypicity(df_data, species_column, species, atypicity, atypicity_column):
    """
    Calcule le pourcentage des
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame contenant au moins les colonnes 'Code_Releve' et la colonne d'atypicité spécifiée
    
    Returns
    -------
    percentage : 
        Pourcentage des observations de l'espèce qui ont une atypicité inférieure à l'observation, arrondi aux dixièmes
    """
    
    df_species = df_data.loc[df_data[species_column]==species, [species_column, atypicity_column]].dropna()
    sum_obs_with_lower_value = sum(df_species[atypicity_column] < atypicity) 
    percentage = round(100*sum_obs_with_lower_value/(len(df_species)-1), 1)
    return percentage

def compute_atypicity(filtered_data, data, method, species_column):
    """
    Calcule un score d'atypicité normalisé sur l'échelle 0-10, et ajoute une colonne "Atypicité" correspondante

    Le calcul actuel normalise la colonne "rank_ground_truth" présente dans
    ``filtered_data`` en utilisant l'étendue (min/max) calculée sur ``data``
    (l'ensemble complet) pour garantir une échelle cohérente entre sous-ensembles.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame contenant les observations filtrées (doit contenir
        ``rank_ground_truth``).
    data : pandas.DataFrame
        DataFrame complet utilisé pour déterminer l'échelle (min/max).
    method : str
        Méthode de calcul. Actuellement pris en charge : ``"rank_ground_truth"``.
    species_column : str
        Nom de la colonne contenant les noms d'espèces dans les DataFrames.
    
    Returns
    -------
    numpy.ndarray
        Tableau 1D (ou Series convertible) contenant le score d'atypicité pour
        chaque ligne de ``filtered_data`` (valeurs entre 0 et 10).

    Notes
    -----
    Si la plage (max-min) vaut 0 (toutes les valeurs identiques), la fonction
    renvoie un vecteur de zéros pour éviter une division par zéro.
    """

    match method:
        case "Atypicité_NFaure":
            minv = np.min(data["rank_ground_truth"])
            maxv = np.max(data["rank_ground_truth"])
            denom = maxv - minv
            if denom != 0:
                return 10 * (filtered_data["rank_ground_truth"].values - minv) / denom
            else:
                return np.zeros(len(filtered_data))

        case "Atypicité_Kohonen":
            minv = np.min(data["RangEspUC"])
            maxv = np.max(data["RangEspUC"])
            denom = maxv - minv
            if denom != 0:
                return 10 * (filtered_data["RangEspUC"].values - minv) / denom
            else:
                return np.zeros(len(filtered_data))
        
        case "Atypicité_Fréquence":
            # TODO: normaliser sur + petite / + grande fréquence observée
            # Return an atypicity score based on species frequency in the whole dataset
            species_counts = data[species_column].value_counts(normalize=True)
            return filtered_data[species_column].map(lambda x: 10 * (1 - species_counts.get(x, 0))).values
        
        case "Atypicité_Hybride":
            print("Méthode 'Atypicité_Hybride' non implémentée.")
            return np.zeros(len(filtered_data))