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
    Calcule un score d'atypicité normalisé sur l'échelle 0-10.

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
    
    Si l'observation n'a ni rank_ground_truth ni RangEspUC, l'atypicité par défaut
    est fixée à 10 quelle que soit la méthode.
    """
    
    # Initialiser le résultat avec la même taille que filtered_data
    atypicity_scores = np.zeros(len(filtered_data))
    
    # Créer un mask pour les observations sans rank_ground_truth ET sans RangEspUC
    has_rank_ground_truth = filtered_data["rank_ground_truth"].notna()
    has_rang_esp_uc = filtered_data["RangEspUC"].notna()
    mask_missing_both = ~has_rank_ground_truth & ~has_rang_esp_uc
    
    # Assigner 10 par défaut pour les observations sans les deux colonnes
    atypicity_scores[mask_missing_both] = 10

    match method:
        case "Atypicité_NFaure":
            mask_valid = has_rank_ground_truth & ~mask_missing_both
            if mask_valid.any():
                minv = np.min(data["rank_ground_truth"])
                maxv = np.max(data["rank_ground_truth"])
                denom = maxv - minv
                if denom != 0:
                    atypicity_scores[mask_valid] = 10 * (filtered_data.loc[mask_valid, "rank_ground_truth"].values - minv) / denom
            return atypicity_scores

        case "Atypicité_Kohonen":
            mask_valid = has_rang_esp_uc & ~mask_missing_both
            if mask_valid.any():
                minv = np.min(data["RangEspUC"])
                maxv = np.max(data["RangEspUC"])
                denom = maxv - minv
                if denom != 0:
                    atypicity_scores[mask_valid] = 10 * (filtered_data.loc[mask_valid, "RangEspUC"].values - minv) / denom
            return atypicity_scores
        
        case "Atypicité_Fréquence":
            # Return an atypicity score based on species frequency in the whole dataset
            # smaller frequency -> higher atypicity
            minv = np.min(data["Frequence"])
            maxv = np.max(data["Frequence"])
            denom = maxv - minv
            if denom != 0:
                return 10 * (1 - (filtered_data["Frequence"].values - minv) / denom)
            else:
                return np.zeros(len(filtered_data))
        
        # case "Atypicité_Hybride":
        #     # Par défaut, calcule un score hybride égal à 50% NFaure + 50% Kohonen,
        #     # en normalisant chaque métrique sur l'échelle 0-10 comme ci-dessus.
        #     # Les pondérations personnalisées sont gérées côté UI (app_annotation).

        #     # NFaure
        #     if ("rank_ground_truth" in filtered_data.columns) and ("rank_ground_truth" in data.columns) and (not filtered_data["rank_ground_truth"].isna().all()):
        #         nf_min = np.nanmin(data["rank_ground_truth"]) if len(data["rank_ground_truth"]) else 0
        #         nf_max = np.nanmax(data["rank_ground_truth"]) if len(data["rank_ground_truth"]) else 0
        #         nf_denom = nf_max - nf_min
        #         nf_scores = 10 * (filtered_data["rank_ground_truth"].values - nf_min) / nf_denom if nf_denom != 0 else np.zeros(len(filtered_data))
        #     else:
        #         nf_scores = np.zeros(len(filtered_data))

        #     # Kohonen
        #     if ("RangEspUC" in filtered_data.columns) and ("RangEspUC" in data.columns) and (not filtered_data["RangEspUC"].isna().all()):
        #         ko_min = np.nanmin(data["RangEspUC"]) if len(data["RangEspUC"]) else 0
        #         ko_max = np.nanmax(data["RangEspUC"]) if len(data["RangEspUC"]) else 0
        #         ko_denom = ko_max - ko_min
        #         ko_scores = 10 * (filtered_data["RangEspUC"].values - ko_min) / ko_denom if ko_denom != 0 else np.zeros(len(filtered_data))
        #     else:
        #         ko_scores = np.zeros(len(filtered_data))

        #     # Hybride 50/50 par défaut
        #     return 0.5 * nf_scores + 0.5 * ko_scores
        
def compute_frequency(data, species_column):
    """
    Calcule une approximation de la fréquence d'apparition de chaque espèce
    dans la base de données : calcule en réalité la fréquence d'apparition de
    chaque espèce dans une version de la base de données réduite aux espèces
    présentes dans le dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame contenant au moins les colonnes ["ID", species_column, "NbObs"].
    species_column : str
        Nom de la colonne contenant les noms d'espèces dans le DataFrame.

    Returns
    -------
    pd.Series
        Série pandas contenant la fréquence d'apparition de l'espèce pour chaque observation
        du dataframe.
    """
    # Nombre total d'observations (NbObs) dans le DataFrame
    total_observations = data["NbObs"].sum()

    # Récupérer le nombre d'observations par espèce
    # Si toutes les valeurs de NbObs pour une espèce sont identiques (ce qui devrait en
    # théorie être le cas), on prend cette valeur.
    # Sinon, on prend la plus grande valeur de NbObs pour cette espèce.
    species_observations = data.groupby(species_column)["NbObs"].max()
    
    # Calcul de la fréquence d'apparition pour chaque observation
    frequency = data[species_column].map(lambda x: species_observations.get(x, 0) / total_observations)

    return frequency