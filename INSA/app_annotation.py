from pydoc import doc
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import pandas as pd
import os
from pathlib2 import Path
from ruamel.yaml import YAML
import numpy as np 
import branca.colormap as cm
import debugpy
from templates import add_template2map
from database_management import get_mEsp_liste
import time
import re
import matplotlib.pyplot as plt

# modules faisant partie du projet
import metrics
import data_utils

"""
Outil d'annotation (Streamlit) pour des observations floristiques.

Ce module fournit :
- le chargement et le filtrage des données
- le calcul d'un score d'atypicité
- la création d'une carte Folium pour visualiser les observations
- l'interface d'annotation et l'export des annotations
- l'affichage des métadonnées associées aux observations
"""

st.set_page_config(layout="wide")

# TODO: renvoyer les chemins d'accès, paramètres, constants, etc. dans un fichier de config séparé (en .yml)
path2param = Path(__file__).parent / "params.yml"

print(f"Loading parameter file: {path2param.resolve()}")
yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
# TODO : fonction cachée
params = yaml.load(path2param)
DATA_PATH_NFAURE = params['DATA_PATH_NFAURE']
print(f"N. Faure's data path: {Path(DATA_PATH_NFAURE).resolve()}\n")

DATA_PATH_KOHONEN = params['DATA_PATH_KOHONEN']
print(f"Kohonen's data path: {Path(DATA_PATH_KOHONEN).resolve()}\n")

VILLARET_PATH = params['VILLARET_PATH']
print(f"Data path: {Path(VILLARET_PATH).resolve()}\n")

FACT_PATH = params['FACT_PATH']
print(f"Data path: {Path(FACT_PATH).resolve()}\n")

species_list_path = params['species_list_path']
print(f"Species list path: {Path(species_list_path).resolve()}\n")

export_path = Path(params['export_path'])
print(f"Export path: {export_path.resolve()}\n")

ISERE = params['ISERE']
__width__ = params['width']
__height__ = params['height']
species_column = params['species_column']


# DATA_PATH_NFAURE = "../../result_export.csv"
# DATA_PATH_KOHONEN = "experiments/kohonen_results.csv"


__GRENOBLE__ = (45.0106, 9.4330)
colormap = cm.LinearColormap(["green", "yellow", "red", "purple"], vmin=0, vmax=10, caption="échelle d'atypicité")


# TODO: ajouter l'inférence et la détermination de rank_ground_truth, en faisant appel à un modèle pré-entraîné
# TODO: ajouter la "Note" d'Alain comme option, ainsi que la fréquence, et proposer un indicateur synthétique. Une classe atypicité pourra être créée.
@st.cache_data
def compute_atypicity_from_metrics(filtered_data, data, method):
    """
    Appelle la fonction metrics.compute_atypicity, qui calcule un score d'atypicité normalisé sur 
    l'échelle 0-10.

    Le calcul normalise la métrique choisie présente dans ``filtered_data`` 
    en utilisant l'étendue (min/max) calculée sur ``data`` (l'ensemble complet) 
    pour garantir une échelle cohérente entre sous-ensembles.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        DataFrame contenant les observations filtrées (doit contenir
        ``rank_ground_truth`` ou ``RangEspUC`` selon la méthode).
    data : pandas.DataFrame
        DataFrame complet utilisé pour déterminer l'échelle (min/max).
    method : str
        Méthode de calcul : "Atypicité_NFaure", "Atypicité_Kohonen",
        "Atypicité_Fréquence" ou "Atypicité_Hybride".

    Returns
    -------
    numpy.ndarray
        Tableau 1D contenant le score d'atypicité pour chaque ligne de 
        ``filtered_data`` (valeurs entre 0 et 10).

    Notes
    -----
    Si la plage (max-min) vaut 0, la fonction renvoie un vecteur de zéros 
    pour éviter une division par zéro.
    """
    return metrics.compute_atypicity(filtered_data, data, method, species_column)


# TODO: charger aussi une liste de milieux "villaret", à des fins d'annotations milieux
# TODO: charger les annotations déjà effectuée en initialisant st.session_state.output_data
@st.cache_data
def load_data(path_nfaure, path_kohonen):
    """
    Charge le jeu de données principal issu de la base Infloris

    Parameters
    ----------
    path_nfaure : string
        chemin vers les données du modèle de N. Faure
    path_kohonen : string
        chemin vers les données du modèle de Kohonen

    Returns
    -------
    data : pandas.DataFrame
        DataFrame contenant l'ensemble des données
        
    observateurs : list
        Liste des observateurs présents dans les données, sans répétition
    
    especes : list
        Liste des espcèces présentes dans les données, sans répétition
    """
    print("Chargement des données...")
    df_nf = pd.read_csv(path_nfaure, sep=";")    
    if "Nom flore" in df_nf.columns:
        df_nf = df_nf.rename(columns={"Nom flore": species_column})
    cols_nf = ["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", 
               species_column, "NbObs", "Nom_Valide", "Groupe", 
               "Date_Releve", "Code_Releve", "NbObs_Releve", "Code_Espèce"]
    df_nf = df_nf[[c for c in cols_nf if c in df_nf.columns]]
    df_nf = df_nf.rename(columns={"Code_Espèce": "Code_Espece"})
    print(f"Taille du dataframe N. Faure : {df_nf.shape}")

    df_ko = pd.read_csv(path_kohonen, sep=",")
    cols_ko = ["PrenomNom", "Lat", "Lon", "RangEspUC", species_column,
               "NbObs", "Nom_Valide", "Groupe", "DateObs", "Code_Releve",
               "NbObs_Releve", "Code_Espece", "Code_Observation", "UC",
               "distUCAvere", "ProbaObs"]
    df_ko = df_ko[[c for c in cols_ko if c in df_ko.columns]]
    df_ko = df_ko.rename(columns={"Lon": "Longitude", "Lat": "Latitude", "DateObs": "Date_Releve"})
    print(f"Taille du dataframe Kohonen : {df_ko.shape}\n")

    # Lister les colonnes communes aux deux data frames
    cols_communes = set(df_nf.columns).intersection(set(df_ko.columns))

    # Colonnes utilisées pour la fusion des deux data frames
    cols_to_merge = ["Code_Releve", "Code_Espece", "Latitude", "Longitude"]

    # Verification des types de données pour les colonnes de fusion
    for col in cols_to_merge:
        if df_nf[col].dtype != df_ko[col].dtype:
            print(f"Avertissement : type de données différent pour la colonne '{col}' : df_nf {df_nf[col].dtype}, df_ko {df_ko[col].dtype}")
            # On tente une conversion explicite si possible
            try:
                df_ko[col] = df_ko[col].astype(df_nf[col].dtype)
                print(f"Conversion réussie de la colonne '{col}' dans df_ko vers {df_nf[col].dtype}\n")
            except Exception as e:
                print(f"Échec de la conversion de la colonne '{col}' dans df_ko : {e}\n")

    # On supprime les lignes auxquelles il manque des données dans les colonnes de fusion
    len_before_nf = len(df_nf)
    df_nf = df_nf.dropna(subset=cols_to_merge)
    len_before_ko = len(df_ko)
    df_ko = df_ko.dropna(subset=cols_to_merge)
    # On print dans le terminal le nombre de lignes supprimées
    print(f"Suppression de {len_before_nf - len(df_nf)} lignes dans les données N. Faure (données incomplètes sur les colonnes de fusion).")
    print(f"Suppression de {len_before_ko - len(df_ko)} lignes dans les données Kohonen (données incomplètes sur les colonnes de fusion).\n")

    # On supprime les lignes qui ont 0 comme code espèce ou 0 comme UC (glaciers)
    len_before_nf = len(df_nf)
    len_before_ko = len(df_ko)
    df_nf = df_nf[df_nf["Code_Espece"] != 0]
    df_ko = df_ko[df_ko["Code_Espece"] != 0]
    df_ko = df_ko[df_ko["UC"] != 0]
    print(f"Suppresion de {len_before_nf - len(df_nf)} lignes dans les données N. Faure (Code_Espece = 0).")
    print(f"Suppresion de {len_before_ko - len(df_ko)} lignes dans les données Kohonen (Code_Espece = 0 ou UC = 0).\n")

    # On supprime les lignes qui ont le même [Code_Releve, Code_Espece] pour ne garder qu'une observation par espèce et par relevé
    len_before_nf = len(df_nf)
    len_before_ko = len(df_ko)
    df_nf = df_nf.drop_duplicates(subset=["Code_Releve", "Code_Espece"])
    df_ko = df_ko.drop_duplicates(subset=["Code_Releve", "Code_Espece"])
    print(f"Suppression de {len_before_nf - len(df_nf)} lignes en double (même Code_Releve et Code_Espece) dans le dataframe N. Faure.")
    print(f"Suppression de {len_before_ko - len(df_ko)} lignes en double (même Code_Releve et Code_Espece) dans le dataframe Kohonen.\n")

    # On print la taille des deux data frames après nettoyage
    print(f"Taille du dataframe N. Faure après nettoyage : {df_nf.shape}")
    print(f"Taille du dataframe Kohonen après nettoyage : {df_ko.shape}\n")

    # Fusion des deux data frames
    print("Fusion des données N. Faure et Kohonen en un unique dataframe...")
    print(f"Colonnes utilisées pour la fusion des données : {cols_to_merge}\n")
    data = pd.merge(df_nf, df_ko, on=cols_to_merge, how="outer", suffixes=("_nf", "_ko"))
    # Pour les colonnes présentes dans la liste cols_communes mais pas dans cols_to_merge, on garde les valeurs de df_ko si elles existent, sinon celles de df_nf
    for col in cols_communes:
        if col not in cols_to_merge:
            data[col] = data[f"{col}_ko"].combine_first(data[f"{col}_nf"])
            data = data.drop(columns=[f"{col}_nf", f"{col}_ko"])
    
    # On rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique : ID = "Code_Releve"_"Code_Espece"
    data["ID"] = data["Code_Releve"].astype(str) + "_" + data["Code_Espece"].astype(str)

    # Ajout d'une colonne "Frequence"
    data["Frequence"] = metrics.compute_frequency(data, species_column)

    # Affichage d'informations sur le nouveau dataframe
    print(f"Données fusionnées : {len(data)} observations.")
    # Nb de données qui ont à la fois rank_ground_truth et Code_Observation => données présentes dans les deux datasets
    count_both_scores = data.dropna(subset=["rank_ground_truth", "Code_Observation"]).shape[0]
    print(f"Dont {count_both_scores} observations présentes dans les deux datasets (N. Faure et Kohonen).\n")
    print(f"Colonnes présentes dans les données fusionnées : {data.columns.tolist()}\n")
    # Affichage du nombre de NaN par colonne
    for col in data.columns:
        num_nan = data[col].isna().sum()
        if num_nan > 0:
            print(f"Colonne '{col}' : {num_nan} valeurs NaN.")
    print("")


    # Calcul des scores d'atypicité
    data["Atypicité_NFaure"] = compute_atypicity_from_metrics(data, data, "Atypicité_NFaure")
    data["Atypicité_Kohonen"] = compute_atypicity_from_metrics(data, data, "Atypicité_Kohonen")
    data["Atypicité_Fréquence"] = compute_atypicity_from_metrics(data, data, "Atypicité_Fréquence")
    # Hybride : par défaut 50/50
    data["Atypicité_Hybride"] = 0.5 * data["Atypicité_NFaure"] + 0.5 * data["Atypicité_Kohonen"]
    
    # # Par défaut pour la carte, on peut créer une colonne 'Atypicité' basée sur le filtre actif
    # if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
    #     method = st.session_state.filters["Méthode"]
    #     if method == "rank_ground_truth":
    #         data["Atypicité"] = data["Atypicité_NFaure"]
    #     elif method == "kohonen":
    #          data["Atypicité"] = data["Atypicité_Kohonen"]
    #     elif method == "frequency":
    #          data["Atypicité"] = data["Atypicité_Frequency"]
    #     else:
    #         st.error(f"Méthode '{method}' non implémentée pour la colonne 'Atypicité'.")
    # else:
    #     data["Atypicité"] = data["Atypicité_Frequency"] # valeur par défaut

    observateurs = list(data["PrenomNom"].unique())

    # Loading species list, with a backup plan in case of failure
    try:
        especes = get_mEsp_liste(species_list_path)
        especes_not_in_data = [espece for espece in data[species_column].unique() if espece not in especes]
        if len(especes_not_in_data) > 0:
            st.write(f"Attention : {len(especes_not_in_data)} espèces présentes dans les données mais absentes de la base d'espèces InFloris. Elles seront ajoutées à la liste des espèces.")
        especes.extend(especes_not_in_data)
    except Exception as e:
        st.write(f"Erreur de chargement de la base d'espèces InFloris (chemin: {species_list_path}). Erreur: {e}")
        st.write("La liste des espèces sera extraite des données chargées.")
        especes = list(np.sort(data[species_column].unique()))

    for i in ["espece", "longitude", "latitude", "micro", "remarque"]:
        if not f"annotation_{i}" in data.columns: # do not erase existing validation annotations
            data[f"annotation_{i}"] = None
    if not "validation" in data.columns: # do not erase existing validation annotations
        data["validation"] = None
    else:
        data["validation"] = data["validation"].fillna(None)
    st.success(f"Données chargées avec succès: {len(data)} observations, {len(observateurs)} observateurs, {len(especes)} espèces.")
    return data, observateurs, especes

@st.cache_data
def load_Villaret(filename):
    """
    Charge les listes de référence d’espèces caractéristiques des milieux isérois, issues de l'ouvrage de Villaret.

    Ces données servent de référence écologique pour la détection de milieux par similarité floristique.

    Parameters
    ----------
    filename : string
        chemin vers les données
        
    Returns
    -------
    dict_milieux : dict
        dictionnaire associant le nom du milieu à son identifiant
    
    milieux_pour_chaque_espece : dict
        dictionnaire associant à chaque espèce la liste des milieux où elle est présente
    
    especes_pour_chaque_milieu : dict
        dictionnaire associant à chaque milieu la liste des espèces qui l'habitent
    """

    df_noms_milieux = pd.read_excel(filename, skiprows=0, header=None, sheet_name="Villaret - fiche>nom", index_col=0, engine="openpyxl").rename(columns={1: "nom"})
    df_milieu_pour_chaque_espece = pd.read_excel(filename, skiprows=3, header=None, sheet_name="Villaret - espèce>fiche", index_col=0, engine="openpyxl").drop([1,2,3], axis=1)
    df_especes_pour_chaque_milieu = pd.read_excel(filename, skiprows=0, header=0, sheet_name="Villaret - fiche>espèces", engine="openpyxl")
    if "Nom flore" in df_milieu_pour_chaque_espece.columns:
        df_milieu_pour_chaque_espece = df_milieu_pour_chaque_espece.rename(columns={"Nom flore": species_column})

    if "Nom flore" in df_milieu_pour_chaque_espece.columns:
        df_especes_pour_chaque_milieu = df_especes_pour_chaque_milieu.rename(columns={"Nom flore": species_column}) 
    
    dict_milieux = dict()
    for i in range(len(df_noms_milieux)):
        dict_milieux[df_noms_milieux.index[i]] = df_noms_milieux.iat[i, 0]
    
    milieu_pour_chaque_espece = dict()
    for i in range(len(df_milieu_pour_chaque_espece)):
        milieu_pour_chaque_espece[df_milieu_pour_chaque_espece.index[i]] = df_milieu_pour_chaque_espece.iloc[i].dropna().to_list()

    especes_pour_chaque_milieu = dict()
    for i in range(len(df_noms_milieux)):
        ID = df_noms_milieux.index[i]
        nom = df_noms_milieux.at[ID, "nom"]
        especes = df_especes_pour_chaque_milieu.loc[2:,ID].dropna().to_list()
        especes_pour_chaque_milieu[ID] = {"nom" : nom, "espèces":especes}

    return dict_milieux, milieu_pour_chaque_espece, especes_pour_chaque_milieu

@st.cache_data
def load_data_fact_abiotiques(filename):
    """
    Charge les données de facteurs abiotiques ou écologiques de référence (indices d'Ellenberg) pour chaque espèce disponible.
    
    Ces données peuvent être utilisées pour enrichir l’interprétation écologique
    des observations et des micro-milieux potentiels.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les facteurs abiotiques ou écologiques associés aux milieux.
    """
    chunk_size = 1_000 # les données seront chargées par paquets pour aller plus vite
    chunks = [] # liste qui contiendra tous les paquets de données

    for chunk in pd.read_csv(filename, sep="\t", header=3, chunksize=chunk_size,
                             usecols=["Nom flore", "Lumiere", "Temperature", "Humidite_edaphique", "Reaction_du_sol_(pH)", "Niveau_trophique", "Matiere_organique", "Salinite", "Texture", "Continentalite"]):
        chunks.append(chunk)
    data_fact = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data_fact = data_fact.dropna()

    if "Nom flore" in data_fact.columns:
        data_fact = data_fact.rename(columns={"Nom flore": species_column})
    
    return data_fact

@st.cache_data
def filter_data(data, filters):
    """
    Filtre les observations selon des critères définis par l’utilisateur à travers l'UI.

    Cette fonction isole un sous-ensemble cohérent du jeu de données principal,
    sans modifier les données originales.

    Parameters
    ----------
    data : pandas data frame
        tableau contenant uniquement les données utiles
        
    filters : dict
        dictionnaire associant une valeur à chaque types de filtre
        
    Returns
    -------
    filtered_data : pandas data frame
        tableau contenant les données filtrées
    """    

    filtered_data = data.copy() # pour l'instant, les données ne sont pas filtrées
    
    if (len(filters["PrenomNom"]) == 0) and (len(filters[species_column]) == 0) and not filters["ID"]: # on vérifie que l'utilisateur a choisi au moins un observateur ou une espèce, sinon il y a trop de données à afficher, et c'est plus logique
        st.error("Veuillez choisir un observateur, une espèce ou un ID.") # erreur qui s'affiche si ce n'est pas le cas
        return filtered_data, False
    else :
        if len(filters["PrenomNom"]) != 0: # si il y a un filtre sur l'observateur
            filtered_data = filtered_data.loc[[element in filters["PrenomNom"] for element in filtered_data["PrenomNom"]]] # on filtre par rapport à l'observateur
        if len(filters[species_column]) != 0: # si il y a un filtre sur l'espèce
            filtered_data = filtered_data.loc[[element in filters[species_column] for element in filtered_data[species_column]]] # on filtre par rapport à l'espèce
        # TODO: ajouter vérification de l'existence de l'ID
        if filters["ID"]:
            filtered_data = filtered_data.loc[filtered_data["ID"]==filters["ID"]]
        if filters["Debut"] > filters["Fin"] : # si le dates choisies ne sont pas dans le bon ordre
            st.error("Veuillez choisir une date de début antérieure à la date de fin.") # erreur affichée
        else :
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date >= filters["Debut"]] # sinon, on garde uniquement les données ultérieures à la date de début choisie
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date <= filters["Fin"]] # puis, on garde uniquement les données précédant la date de début choisie
        
        # filtered_data["Atypicité"] = compute_atypicity_from_metrics(filtered_data, data, filters["Méthode"])
        filtered_data = filtered_data.loc[filtered_data[filters["Méthode"]]<=filters["hi_Score"]]
        filtered_data = filtered_data.loc[filtered_data[filters["Méthode"]]>=filters["lo_Score"]]
        filtered_data = filtered_data.sort_values(by=filters["Méthode"], ascending=False).head(int(filters['Top_atypicity']))
        return filtered_data, True 


def compute_center(data):
    """
    Calcule le centre géographique des données pour l'affichage de la carte

    Parameters
    ----------
    data : pandas data frame
        tableau contenant les données filtrées
        
    Returns
    -------
    center : tuple
        coordonnées (latitude, longitude) du centre géographique des données
    """  
    lat_center = float(data["Latitude"].max() + data["Latitude"].min())/2
    lon_center = float(data["Longitude"].max() + data["Longitude"].min())/2
    lat_expand = data["Latitude"].max() - data["Latitude"].min()
    lon_expand = data["Longitude"].max() - data["Longitude"].min()
    expand = max(lat_expand, lon_expand)
    if expand < 0.1: zoom = 11
    elif expand < 0.3: zoom = 10
    elif expand < 1: zoom = 9
    else: zoom = 8
    return (lat_center, lon_center), zoom


# @st.cache_data
def make_map(df, colormap, toggle_clusters=False, toggle_dpt=False, annotated=[],
             center=__GRENOBLE__, zoom_start=8):
    """Créer une carte Folium contenant les observations filtrées.

    La fonction crée une instance de :class:`folium.Map`, y ajoute éventuellement
    la couche GeoJSON du département si `st.session_state.dpt` est vrai, puis
    construit un :class:`folium.FeatureGroup` nommé "observations" et y ajoute
    des ``CircleMarker`` pour chaque ligne de ``df``. Les marqueurs ont des
    tailles différentes selon qu'une observation est dans la liste
    ``annotated`` (petit) ou qu'elle correspond à l'identifiant sélectionné
    stocké dans ``st.session_state.id_obs`` (plus grand).

    Paramètres
    ----------
    df : pandas.DataFrame
        DataFrame contenant les observations filtrées. Doit contenir au moins
        les colonnes : ``Latitude``, ``Longitude``, ``ID``, ``Nom flore``,
        ``Atypicité``.
    colormap : branca.colormap
        Colormap utilisée pour déterminer la couleur de remplissage des marqueurs
        en fonction de la valeur d'atypicité.
    toggle_clusters : bool, optional
        Si True, les marqueurs sont ajoutés à un ``MarkerCluster`` (par défaut
        False).
    toggle_dpt : bool, optional
        Option prévue pour afficher le département (non utilisée directement
        car l'affichage du département se fait via ``st.session_state.dpt``).
    annotated : list, optional
        Liste d'IDs d'observations déjà annotées (affichées avec un rayon réduit).
    center : tuple, optional
        Coordonnées (lat, lon) utilisées comme centre de la carte si le
        calcul automatique échoue (par défaut ``__GRENOBLE__``).
    zoom_start : int, optional
        Niveau de zoom initial de la carte.

    Returns
    -------
    folium.Map, folium.FeatureGroup
        L'objet ``folium.Map`` créé et le ``FeatureGroup`` contenant les
        marqueurs.

    Notes
    -----
    Cette fonction ne doit pas être décorée avec ``@st.cache_data`` car les
    objets Folium ne sont pas sérialisables par le cache de Streamlit.
    """

    try:
        center, zoom_start = compute_center(df)
    except:
        pass

    map_ = folium.Map(location=center, zoom_start=zoom_start) #affiche la carte centrée sur Grenoble
    if st.session_state.dpt: 
        folium.GeoJson(ISERE).add_to(map_)   
    
    group_1 = folium.FeatureGroup("observations")

    for index, row in df.iterrows(): # on affiche les N marqueurs
        if row['ID'] in annotated: radius = 3
        else: radius = 7

        if np.isnan(row[st.session_state.filters["Méthode"]]):
            fill_c = "white"
        else:
            fill_c = colormap(float(row[st.session_state.filters["Méthode"]]))

        folium.CircleMarker(location=list(row.loc[['Latitude', 'Longitude']]),
                            radius=radius,
                            color="black",
                            fill=True,
                            fill_color=fill_c,
                            fill_opacity=1,
                            popup=f"ID : {row.ID}<br> espèce : {row[species_column]}<br> atypicité : {round(row.loc[st.session_state.filters["Méthode"]], 3)}", 
                            tooltip=f"ID : {row.ID}<br> espèce : {row[species_column]}<br> atypicité : {round(row.loc[st.session_state.filters["Méthode"]], 3)}"
                            ).add_to(group_1)

    if toggle_clusters : # affichage groupé des observations
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1.add_to(marker_cluster)
    else :
        group_1.add_to(map_)


    return map_, group_1
    # return st_folium(map_, width=width, height=height, key=key, on_change=callback)


def update_id_obs(st_data, filtered_data, current, last, type_annotation):
    """
    Met à jour l'ID de l'observation sélectionnée à partir des données
    d'évènement renvoyées par le composant Folium / Streamlit-Folium.

    La fonction gère plusieurs formats possibles renvoyés par le composant :
    - ``last_object_clicked_popup`` : texte du popup contenant "ID : <n>"
    - ``last_object_clicked`` : dictionnaire avec un champ ``id``
    - ``last_object_clicked`` : dictionnaire avec ``lat`` et ``lng`` — on recherche
      alors le marqueur le plus proche parmi ``filtered_data``.

    Parameters
    ----------
    st_data : dict
        Données issues du composant Folium (clés possibles listées ci-dessus).
    current : str | None
        ID actuellement sélectionné (format: "Code_Releve_Code_Espece").
    last : str | None
        ID précédemment sélectionné (utilisé pour historique/back-navigation).

    Returns
    -------
    tuple (new_current, new_last)
        Tuples d'IDs mis à jour. Si aucun changement détecté, retourne (current, last).
    """




    new = None
    popup = st_data.get('last_object_clicked_popup') if isinstance(st_data, dict) else None
    clicked = st_data.get('last_object_clicked') if isinstance(st_data, dict) else None

    if popup:
        m = re.search(r'ID\s*:\s*(\d+_\d+)', popup)
        if m:
            try:
                new = m.group(1).strip()
            except:
                new = None
    elif clicked and type_annotation!="coords":
        if isinstance(clicked, dict):
            if 'id' in clicked:
                try:
                    new = str(clicked['id'])
                except:
                    new = None
            elif ('lat' in clicked) and ('lng' in clicked):
                try:
                    lat = float(clicked['lat'])
                    lng = float(clicked['lng'])
                    df = filtered_data
                    if df is not None and len(df) > 0:
                        distances = (df['Latitude'] - lat)**2 + (df['Longitude'] - lng)**2
                        nearest_idx = distances.idxmin()
                        new = str(df.at[nearest_idx, 'ID'])
                except:
                    new = None

    if new is None:
        return (current, last)
    else:
        if new == current:
            return (current, last)
        else:
            return (new, current)

@st.cache_data
def afficher_metadonnees(data, id_obs, output_data):
    """
    Affiche, via Streamlit, les métadonnées et le statut d'annotation d'une
    observation sélectionnée.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les observations (index attendu égal à l'ID).
    id_obs : str
        Identifiant de l'observation à afficher (format: "Code_Releve_Code_Espece").
    output_data : pandas.DataFrame
        Table des annotations sauvegardées (permet d'afficher le statut de
        validation et les corrections déjà enregistrées).

    Notes
    -----
    La fonction construit une liste de lignes préformatées (avec du Markdown)
    et l'affiche en une seule fois pour éviter un espacement vertical excessif.
    """
    
    # Sélection robuste sur la colonne ID (l'index n'est pas l'ID)
    mask = data['ID'] == id_obs
    if not mask.any():
        st.error(f"ID inconnu dans les données filtrées : {id_obs}")
        return
    row = data.loc[mask].iloc[0]

    lines = []
    lines.append(f"**ID :** {id_obs}")

    mask_out = output_data['ID'] == id_obs if 'ID' in output_data else pd.Series([], dtype=bool)
    if mask_out.any():
        lines.append(":green[**observation annotée**]")

        validation_value = output_data.loc[mask_out, 'validation'].dropna().iloc[-1] if not output_data.loc[mask_out, 'validation'].dropna().empty else None
        if validation_value == "Je confirme":
            color = "green"
        elif validation_value == "Donnée douteuse":
            color = "orange"
        elif validation_value is not None:
            color = "red"
        else:
            color = "grey"
        lines.append(f"**Statut:** :{color}[{validation_value if validation_value is not None else 'Non renseigné'}]")

        
        annotations_espece = output_data.loc[mask_out, 'annotation_espece'].dropna()
        annotation_espece = annotations_espece.iloc[-1] if not annotations_espece.empty else None

        if annotation_espece and annotation_espece != row[species_column]:
            lines.append(f"**Espèce ({species_column}):** {row[species_column]} :green[→ {annotation_espece}]")
        else:
            lines.append(f"**Espèce ({species_column}):** {row[species_column]} :green[espèce non modifiée]")

        lines.append(f"**Groupe :** {row['Groupe']}")
        lines.append(f"**Observateur :** {row['PrenomNom']}")
        lines.append(f"**Date :** {pd.to_datetime(row['Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")

        annotations_coords = output_data.loc[mask_out, ['annotation_latitude', 'annotation_longitude']].dropna()
        annotation_coords = annotations_coords.iloc[-1] if not annotations_coords.empty else None

        if isinstance(annotation_coords, pd.Series) and (annotation_coords['annotation_latitude'] != row['Latitude'] or annotation_coords['annotation_longitude'] != row['Longitude']):
            lines.append(f"**Coordonnées :** ({row['Latitude']}, {row['Longitude']}) :green[→ ({annotation_coords['annotation_latitude']}, {annotation_coords['annotation_longitude']})]")
        else:
            lines.append(f"**Coordonnées :** ({row['Latitude']}, {row['Longitude']}) :green[position non modifiée]")
    
    else:
        lines.append(":red[**observation en attente d'annotation**]")
        lines.append(f"**Espèce ({species_column}):** {row[species_column]}")
        lines.append(f"**Groupe :** {row['Groupe']}")
        lines.append(f"**Observateur :** {row['PrenomNom']}")
        lines.append(f"**Date :** {pd.to_datetime(row['Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")
        lines.append(f"**Coordonnées :** ({row['Latitude']}, {row['Longitude']})")

    if pd.notna(row.get('NbObs')):
        lines.append(f"**Spécimens observés :** {int(row['NbObs'])}")
    
    # Affichage des scores d'atypicité comparés
    score_nf = round(row['Atypicité_NFaure'], 3) if pd.notna(row.get('Atypicité_NFaure')) else np.nan
    score_ko = round(row['Atypicité_Kohonen'], 3) if pd.notna(row.get('Atypicité_Kohonen')) else np.nan
    score_freq = round(row['Atypicité_Fréquence'], 3) if pd.notna(row.get('Atypicité_Fréquence')) else np.nan
    lines.append("**Scores d'atypicité :**")
    lines.append(f"- N. Faure : {f'{score_nf} / 10' if not pd.isna(score_nf) else 'N/A'}")
    lines.append(f"- Kohonen : {f'{score_ko} / 10' if not pd.isna(score_ko) else 'N/A'}")
    lines.append(f"- Basé sur la fréquence : {f'{score_freq} / 10' if not pd.isna(score_freq) else 'N/A'}")
    
    if mask_out.any():

        annotations_micro = output_data.loc[mask_out, 'annotation_micro'].dropna()
        annotation_micro = annotations_micro.iloc[-1] if not annotations_micro.empty else None

        if annotation_micro:
            lines.append(f"**Micro-milieu :** :green[{annotation_micro}]")
        else:
            lines.append("**Micro-milieu :** :green[aucun micro-milieu signalé]")


        annotations_remarque = output_data.loc[mask_out, 'annotation_remarque'].dropna()
        annotation_remarque = annotations_remarque.iloc[-1] if not annotations_remarque.empty else None

        if annotation_remarque:
            lines.append(f"**Remarque :** :green[{annotation_remarque}]")
        else:
            lines.append("**Remarque :** :green[aucune remarque]")


    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

# def show_annotations():
#     annotations_remarque = output_data.loc[mask_out, 'annotation_remarque'].dropna()
#     annotation_remarque = annotations_remarque.iloc[-1] if not annotations_remarque.empty else None

#     if annotation_espece and annotation_espece != row[species_column]:
#         lines.append(f"**Remarque :** {row[species_column]} :green[→ {annotation_remarque}]")
#     else:
#         lines.append(f"**Remarque :** {row[species_column]} :green[aucune remarque]")

    

@st.cache_data
def afficher_stats_geo(data, id_obs, output_data):
    """
    Affiche, via Streamlit, les données géographiques associées à une observation, notamment ses coordonnées.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les observations (index attendu égal à l'ID).
    id_obs : int
        Identifiant de l'observation à afficher.
    output_data : pandas.DataFrame
        Table des annotations sauvegardées (permet d'afficher le statut de
        validation et les corrections déjà enregistrées).

    Notes
    -----
    La fonction construit une liste de lignes préformatées (avec du Markdown)
    et l'affiche en une seule fois pour éviter un espacement vertical excessif.
    """
    
    lines = []
    lines.append(f"**ID** : {id_obs}")
    lines.append(f"**coordonnées** : ({row['Latitude']}, {row['Longitude']})")
    # lines.append(f"**commune** : {'À implémenter'}")
    
    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)
    
    
@st.cache_data
def afficher_stats_atypicite(data, id_obs, output_data, method):
    """
    Affiche les informations de fréquence et d'atpicité associées à une observation et à son espèce.

    Inclut des informations telles que le nombre de spécimens observés, la fréquence ou le pourcentage 
    des spécimens qui sont moins atypiques que l'observation sélectionnée.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les observations (index attendu égal à l'ID).
    id_obs : int
        Identifiant de l'observation à afficher.
    output_data : pandas.DataFrame
        Table des annotations sauvegardées (permet d'afficher le statut de
        validation et les corrections déjà enregistrées).

    Notes
    -----
    La fonction construit une liste de lignes préformatées (avec du Markdown)
    et l'affiche en une seule fois pour éviter un espacement vertical excessif.
    """
    
    lines = []
    lines.append(f"**spécimens observés** : {int(row['NbObs'])}")
    # lines.append(f"**fréquence de l'espèce** : {float(row['Frequence_espece'][1:-1:])}")
    lines.append(f"**atypicité** : {round(row[method], 3)}")
    
    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

@st.cache_data
def afficher_stats_releve(data, id_obs):
    """
    Affiche des informations complémentaires sur le relevé auquel
    appartient l'observation sélectionnée : code et date du relevé, nombre d'observations, observateur.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les observations.
    id_obs : int
        Identifiant de l'observation.
    """
    mask = data['ID'] == id_obs
    if not mask.any():
        st.error(f"ID inconnu dans les données filtrées : {id_obs}")
        return
    row = data.loc[mask].iloc[0]

    lines = []
    lines.append(f"**observateur** : {row['PrenomNom']}")
    lines.append(f"**code relevé** : {row['Code_Releve']}")
    lines.append(f"**date** : {data_utils.date_francaise(row['Date_Releve'])}")
    lines.append(f"**observations dans le relevé** : {int(row['NbObs_Releve'])}")
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

def save_annotations(data, export_path):
    """
    Exporte la table d'annotations vers un fichier CSV horodaté.

    Le fichier de sortie est créé à partir de ``export_path`` en ajoutant
    un suffixe de date/heure pour éviter d'écraser les exports précédents.

    Parameters
    ----------
    data : pandas.DataFrame
        Table des annotations à exporter.
    export_path : pathlib.Path
        Chemin de base pour l'export (le timestamp sera ajouté au nom).
    """
    export_path = export_path.parent / (export_path.stem + time.asctime().replace(" ", "_").replace(":", "-") + export_path.suffix)
    data.to_csv(export_path, sep=";", index=False)
    st.success(f"Données exportées vers {export_path.resolve()}")

def _save_annotation(id_obs, validation_key, type_annotation):
    """
    Sauvegarde en mémoire (session) l'annotation d'une observation sélectionnée.

    Cette fonction lit l'état des widgets liés à l'observation (clé de
    validation et sélection d'espèce) dans ``st.session_state`` et met à jour
    ``st.session_state.output_data`` en conservant la dernière valeur par ID.

    Parameters
    ----------
    id_obs : int
        ID de l'observation à sauvegarder.
    validation_key : str
        Clé utilisée dans ``st.session_state`` pour lire l'état du widget
        de validation (par ex. "validation_<id>").

    Notes
    -----
    - La fonction s'appuie sur la variable globale ``data`` pour récupérer la
      ligne correspondant à l'ID (cette variable est définie au niveau module
      lors du chargement initial).
    - Si ``st.session_state.output_data`` n'existe pas encore, elle l'initialise.
    """
    if type_annotation is None : 
        st.error("Veuilez choisir une option d'annotation")
        return

    if id_obs is None:
        st.error("Aucune observation sélectionnée — impossible de sauvegarder.")
        return

    if 'output_data' not in st.session_state:
        st.session_state.output_data = pd.DataFrame(columns=data.columns)

    # read the current validation status from session state using the provided key
    validation_status = st.session_state.get(validation_key, None)

    # Get the currently selected annotation for this observation (if any)
    select_key = f"select_{type_annotation}_{id_obs}"
    new_annotation = st.session_state.get(select_key, None)

    if id_obs in list(st.session_state.output_data["ID"]):
        row = st.session_state.output_data.loc[st.session_state.output_data["ID"]==id_obs].copy()
    else:
        row = data.loc[data["ID"]==id_obs].copy()
    
    if type_annotation == "coords":
        new_annotation = new_annotation.replace("(","").replace(")","").split(",")
        cols = ['latitude', 'longitude']

        for i in range(2):
            annotation_col = 'annotation_' + cols[i]

            # ensure annotation_col column exists
            if annotation_col not in row.columns:
                row[annotation_col] = None
            if new_annotation is not None:
                row.loc[:, annotation_col] = float(new_annotation[i].strip())
            # row = row.assign(validation=validation_status)

            # st.session_state.output_data = pd.concat([
            #     st.session_state.output_data,
            #     row
            # ], ignore_index=True).drop_duplicates(subset=['ID'], keep='last')

    else : 
        annotation_col = 'annotation_' + type_annotation

        # ensure annotation_col column exists
        if annotation_col not in row.columns:
            row[annotation_col] = None
        if new_annotation is not None:
            row.loc[:, annotation_col] = new_annotation

        
    row = row.assign(validation=validation_status)

    st.session_state.output_data = pd.concat([
        st.session_state.output_data,
        row
    ], ignore_index=True).drop_duplicates(subset=['ID'], keep='last')

    st.success(f"Annotation sauvegardée pour l'observation ID {id_obs}.")


@st.cache_data    
def get_releve(data, ID_releve):
    """
    Récupère les données associées à un relevé donné.
    Un Dataframe contenant l'ensemble des données du relevé est créé, ainsi 
    qu'un set ne contenant que les noms des espèces présentes.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame contenant l’ensemble des observations.
        
    ID_releve : int
        Code du relevé.
        
    Returns
    -------
    df_releve : pandas.DataFrame
        DataFrame contenant les observations correspondant au relevé sélectionné.
        
    set_releve : set
        Set contenant les espèces du relevé
    """
    set_releve = set()
    df_releve = data.loc[data["Code_Releve"]==ID_releve]
    
    for esp in df_releve["Nom_Valide"]:
        esp=data_utils.format_species(esp)
        set_releve.add(esp)
        
    return df_releve, set_releve

@st.cache_data
def get_set_milieu(set_releve, dict_m_pour_e):
    """
    Crée un set contenant les codes d'identification des milieux qui ont au moins une observation en commun avec le relevé.
    Cette fonction est utilisée pour le calcul de l'indice de Jaccard.
    
    Parameters
    ----------
    set_releve : set
        Espèces observées dans le relevé
        
    dict_m_pour_e : dict
        Dictionnaire associant à une espèce les milieux dans lesquels elle est 
        présente, selon l'ovrage de Villaret
        
    Returns
    -------
    set_milieu : set
        Set contenant les milieux qui donneront un indice de Jaccard non nul
    """
    set_milieu = set()
    for esp in set_releve:
        if esp in dict_m_pour_e.keys():
            for m in dict_m_pour_e[esp]:
                if len(m)==5: #TODO : fix F090
                    set_milieu.add(m) 
    return set_milieu

@st.cache_data
def get_df_Jaccard(set_r, set_m, e_pour_m, dict_milieux):
    """
    Construit un DataFrame de scores de similarité de Jaccard pour plusieurs milieux.
    Permet de comparer un relevé à plusieurs références écologiques.
    
    Parameters
    ----------
    set_r : set
        Espèces observées dans le relevé.
        
    set_m : set
        Milieux ayant au moins une espèce en commun avec le relevé.
        
    e_pour_m : dict
        Dictionnaire associant à chaque milieu de référence les espèces caractéristiques qui le définissent.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les scores de similarité par milieu, ainsi que le nombre d'espèces permettant de relier le relevé à chaque milieu.
    """
    list_res = list()
    
    for mil in set_m:
        i_j = metrics.indice_jaccard(set_r, set(e_pour_m[mil]['espèces']))
        list_res.append([dict_milieux[mil], i_j[1], i_j[0]])
    
    df_J = pd.DataFrame(list_res, 
                        index=list(set_m), 
                        columns=["Nom", "Indice de Jaccard", "nbr espèces observées"])
    
    return df_J.sort_values(by="Indice de Jaccard", axis=0, ascending=False)

@st.cache_data
def get_df_fact(df_f, df_r, cols, species_column):
    """
    Rajoute les facteurs écologiques (indices d'Ellenberg) au DataFrame des observatoins du relevé.
    Les facteurs disponibles sont  : ["Lumiere", "Temperature", "Humidite_edaphique", "Reaction_du_sol_(pH)", "Niveau_trophique", "Matiere_organique", "Salinite", "Texture", "Continentalite"]
    
    Parameters
    ----------
    df_f : pandas.DataFrame
        DataFrame contenant les indices d'Ellenberg disponibles pour un ensemble d'espèces.
        
    df_r : pandas.DataFrame
        DataFrame contenant les données du relevé, sans les indices d'Ellenberg
    
    cols : list
        Liste contenant les indices d'Ellenberg à utiliser
        
    species_column : str
        Nom de la colonne contenant les noms valides des espèces
    
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant les données du relevé avec les indices d'Ellenberg.
        
    """
    df_merge = pd.merge(df_f, df_r,how='inner', on=species_column)
    return_cols = [species_column, "Code_Releve"] + cols
    return df_merge[return_cols]
    

@st.cache_data
def get_intra_inter_atypicity(df_data, Code_Releve, method):
    """
    Prépare les données d'atypicité avant de pouvoir les représenter sous forme de boxplot.
    
    Parameters
    ----------
    df_data : pandas.DataFrame
        DataFrame contenant l'ensemble des données
        
    Code_Releve : str
        Identifiant du relevé.
    
    Returns
    -------
    profil_intra : pandas.Series
        Atypicités du relevé.
        
    profil_intra : pandas.Series
        Atypicités moyennes de tous les relevés.
        
    """
    
    profil_intra = df_data.loc[df_data["Code_Releve"]==Code_Releve, method].dropna()
    profil_inter = metrics.compute_mean_atypicity_per_releve(df_data, method)
    
    return profil_intra, profil_inter


def check_filtered_and_clicked(bool_filtered, id_obs):
    if not bool_filtered:
        st.write("Veuillez filtrer les données")
        return False

    elif id_obs is None: # si aucune observation n'a ete selectionnee
        st.write("Veuillez cliquer sur une observation pour afficher ses métadonnées")
        return False
    
    else: 
        return True
    
def get_default_annotation(filtered_data, type_annotation, id_obs, list_options):
    """
    type_annotation parmi ["espece", "longitude", "latitude", "micro", "remarque"]
    
    """
    col_annotation = "annotation_" + type_annotation
    
    mask = filtered_data['ID'] == id_obs
    if not mask.any():
        st.error(f"ID inconnu dans les données filtrées : {id_obs}")
        return
    else :
        row = filtered_data.loc[mask].iloc[0]
        
    # Prefer previously saved annotation (in output_data) if present, otherwise use recorded species
    default_espece = None
    if hasattr(st.session_state, "output_data") and not st.session_state.output_data.empty:
        prev = st.session_state.output_data.loc[st.session_state.output_data['ID']==id_obs, col_annotation]
        if not prev.empty and pd.notna(prev.iloc[-1]):
            default_espece = prev.iloc[-1]
        
    
    if default_espece is None and (filtered_data is not None) and (id_obs in list(filtered_data['ID'])):
        if row[col_annotation] and pd.notna(row[col_annotation]):
            default_espece = row[col_annotation]
        else:
            default_espece = row[species_column] if id_obs in list(filtered_data['ID']) else None

    if default_espece in list_options:
        default_index = list_options.index(default_espece)
    else:
        default_index = 0
        
    return default_index, default_espece


def get_default_annotation(filtered_data, type_annotation, id_obs, list_options):
    """
    type_annotation parmi ["espece", "longitude", "latitude", "micro", "remarque"]
    
    """
    col_annotation = "annotation_" + type_annotation

    mask = filtered_data['ID'] == id_obs
    if not mask.any():
        st.error(f"ID inconnu dans les données filtrées : {id_obs}")
        return
    else :
        row = filtered_data.loc[mask].iloc[0]

    # Prefer previously saved annotation (in output_data) if present, otherwise use recorded species
    default_espece = None
    if hasattr(st.session_state, "output_data") and not st.session_state.output_data.empty:
        prev = st.session_state.output_data.loc[st.session_state.output_data['ID']==id_obs, col_annotation]
        if not prev.empty and pd.notna(prev.iloc[-1]):
            default_espece = prev.iloc[-1]


    if default_espece is None and (filtered_data is not None) and (id_obs in list(filtered_data['ID'])):
        if row[col_annotation] and pd.notna(row[col_annotation]):
            default_espece = row[col_annotation]
        else:
            default_espece = row[species_column] if id_obs in list(filtered_data['ID']) else None

    if default_espece in list_options:
        default_index = list_options.index(default_espece)
    else:
        default_index = 0

    return default_index, default_espece

if __name__ == "__main__":

    tab1, tab2 = st.tabs(["Visu&Annotation", "Statistiques"])

    ###################################################
    # Chargement des donnees
    # msg = st.toast("Chargement des données...")
    data, observateurs, especes = load_data(DATA_PATH_NFAURE, DATA_PATH_KOHONEN)
    dict_milieux, milieu_pour_chaque_espece, especes_pour_chaque_milieu = load_Villaret(VILLARET_PATH)
    data_fact_abiotiques = load_data_fact_abiotiques(FACT_PATH)

    if "output_data" not in st.session_state:
        st.session_state.output_data =  pd.DataFrame(columns=data.columns)


    if "map_center" not in st.session_state:
        st.session_state["map_center"] = __GRENOBLE__
        st.session_state["map_zoom"] = 8

    if "filtered" not in st.session_state:
        st.session_state.filtered = False
        st.session_state.id_obs = None
        st.session_state.last = None
        
    if "filters" in st.session_state:
        filtered_data, st.session_state.filtered = filter_data(data, st.session_state.filters)

    else:
        filtered_data = None
        
    if "last_clicked" not in st.session_state : 
        st.session_state.clicked = None
        st.session_state.last_clicked = None
        st.session_state.type_annotation = None

        
    ###################################################
    # Premier onglet pour la visualisation

    with tab1:
        col_carte, col_annot, col_meta = st.columns([5, 3, 2], border= True, gap=None) # separation de l'affichage en 3 : une partie pour la carte, une pour l'annotation et une pour les metadonnees

        ###################################################
        # Selection des filtres
        
        st.sidebar.subheader("Filtres") # menu de selection des filtres
        with st.sidebar.form(key="filtres"):
            filters = dict()
            filters["PrenomNom"] = st.multiselect("Nom de l'observateur", sorted(observateurs), placeholder="Aucune sélection")
            filters[species_column] = st.multiselect("Espèce", sorted(especes), placeholder="Aucune sélection")
            filters["ID"] = st.text_input("ID", value=None, help="Format : CodeReleve_CodeEspece (ex : 12345_56789)")
            filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            filters["lo_Score"], filters["hi_Score"] = st.select_slider("Atypicité", options=[i for i in np.arange(0, 10.5, 0.5)], value=(0,10))
            filters['Top_atypicity'] = st.slider('Filter les plus atypiques', min_value=5, max_value=100, value=20, step=5)
            # st.markdown('''0 :green[----------]:yellow[----------]:orange[----------]:red[----------]:violet[----------] 10''') # légende
            filters["Méthode"] = st.radio("Méthode de calcul de l'atypicité :", ["Atypicité_NFaure", "Atypicité_Kohonen", "Atypicité_Fréquence", "Atypicité_Hybride"])

            # Slider de pondération pour la méthode hybride
            if filters["Méthode"] == "Atypicité_Hybride":
                default_pct = int(100 * float(st.session_state.get("hybrid_weight_nfaure", 0.5)))
                pct_nf = st.slider("Poids NFaure (%)", min_value=0, max_value=100, value=default_pct)
                filters["hybrid_weight_nfaure"] = pct_nf / 100.0
                filters["hybrid_weight_kohonen"] = 1.0 - filters["hybrid_weight_nfaure"]
                st.caption(f"Pondération: NFaure {pct_nf}% | Kohonen {100 - pct_nf}%")
            
            
            submitted = st.form_submit_button(label="Enregistrer") # validation des filtres
            if submitted : #creation d'un subset des donnees filtrees
                with st.sidebar.status("Selection des données...") as status:
                    # Applique immédiatement la pondération si méthode hybride
                    if filters["Méthode"] == "Atypicité_Hybride":
                        w_nf = float(filters.get("hybrid_weight_nfaure", st.session_state.get("hybrid_weight_nfaure", 0.5)))
                        w_ko = 1.0 - w_nf
                        data["Atypicité_Hybride"] = w_nf * data["Atypicité_NFaure"] + w_ko * data["Atypicité_Kohonen"]
                        # mémorise la pondération dans la session
                        st.session_state.hybrid_weight_nfaure = w_nf
                    
                    filtered_data, st.session_state.filtered = filter_data(data, filters)
                    status.update(label='Données filtrées', state = "complete")

                # with st.sidebar.status("Calcul de l'atypicité sur les données filtrées...") as status:
                #     filtered_data["Atypicité"] = compute_atypicity_from_metrics(filtered_data, 
                #                                                                     data, filters["Méthode"])
                #     status.update(label='Atypicité calculée', state = "complete")
                    
                st.session_state.filters = filters
                st.session_state.id_obs = None
                
        clusters = st.sidebar.toggle("Affichage groupé")
        st.session_state.dpt = st.sidebar.toggle("Afficher le département de l'Isère", 
                                                value=True)


        ###################################################
        # Affichage de la carte
        with col_carte:
            sub_col_carte_1, sub_col_carte_2 = st.columns(2)
            
            sub_col_carte_1.subheader("Carte des observations", help="La couleur d'un point représente son atypicité : du vert pour les valeurs les plus faibles, jusqu'au violet pour les valeurs plus élevées.")
            
            if not st.session_state.filtered:
                st.write("Veuillez filtrer les données")
            elif len(filtered_data) == 0:
                st.error("Aucune observation ne correspond à ces critères. Veuillez essayer avec une autre méthode de calcul de l'atypicité ou en élargissant les filtres.")
            else :
                map1, group1 = make_map(filtered_data,
                                colormap,
                                annotated= st.session_state.output_data['ID'].to_list(),
                                toggle_clusters=clusters, 
                                toggle_dpt=st.session_state.dpt)

                if st.session_state.last_clicked is not None:
                    folium.Marker(
                                location=st.session_state.last_clicked,
                                icon=folium.Icon(color="red"),
                                popup="Marker sélectionné"
                            ).add_to(map1)

                st_data1 = st_folium(map1, key='map1', 
                                     width=__width__)
        
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, filtered_data, st.session_state.id_obs, st.session_state.last, st.session_state.type_annotation)


        #########################
        # Formulaire d'annotation
        with col_annot:
            st.subheader("Formulaire d'annotation")    
            if check_filtered_and_clicked(st.session_state.filtered, st.session_state.id_obs):
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, filtered_data, st.session_state.id_obs, st.session_state.last, st.session_state.type_annotation)
                st.button("Exporter les annotations", on_click=lambda: save_annotations(st.session_state.output_data, export_path))
     
                form_key = f"annotation_{st.session_state.id_obs}"
                validation_key = f"validation_{st.session_state.id_obs}"


                actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Modifier la position", "Signaler un micro-milieux", "Autre (ajouter une remarque)"]
                st.session_state.type_annotation = st.selectbox("Que souhaitez-vous faire ?", actions_possibles, index=None, placeholder="Veuillez choisir une option")
                # annoter(data, action, st.session_state.id_obs, especes)

                with st.form(key=form_key):
                    st.subheader("Annotation de l'observation")
                    # Update the selected id from the map component early so it is preserved across reruns

                    id_obs = st.session_state.id_obs

                    match st.session_state.type_annotation:
                        case "Modifier l'espèce/le nom de l'espèce":
                            if st.session_state.clicked is not None or st.session_state.last_clicked is not None:
                                st.session_state.clicked = None
                                st.session_state.last_clicked = None
                                st.rerun()
                            st.session_state.type_annotation = "espece"
                            select_key = f"select_espece_{st.session_state.id_obs}"
                            default_index, default_espece = get_default_annotation(filtered_data, st.session_state.type_annotation, id_obs, especes)
                            st.selectbox(f"Modifier l'espèce (Valeur initiale: {default_espece})", especes, index=default_index, key=select_key)
                    
                    
                        case "Signaler un micro-milieux":
                            if st.session_state.clicked is not None or st.session_state.last_clicked is not None:
                                st.session_state.clicked = None
                                st.session_state.last_clicked = None
                                st.rerun()
                            st.session_state.type_annotation = "micro"
                            select_key = f"select_micro_{st.session_state.id_obs}"
                            default_index, default_milieu = get_default_annotation(filtered_data, st.session_state.type_annotation, id_obs, list(dict_milieux.values()))
                            st.selectbox(f"Signaler un micro-milieu (Valeur initiale: {'default_milieu'})", np.sort(list(dict_milieux.values())), index=default_index, key=select_key)
                    
                        case "Modifier la position":
                            st.session_state.type_annotation = "coords"
                            select_key = f"select_coords_{st.session_state.id_obs}"
                            st.session_state.clicked = st_data1.get('last_clicked') if isinstance(st_data1, dict) else None

                            st.write("Clickez sur la carte pour choisir la nouvelle position de l'observation")
                            if st.session_state.clicked is None:

                                if st.session_state.last_clicked is None :
                                    st.write("Aucune position selectionnée")

                                else :
                                    st.write("Les nouvelles coordonnées seront : ")
                                    st.text_area("Les nouvelles coordonnées seront : ", value=st.session_state.last_clicked, key=select_key, height="content", disabled=True, label_visibility="collapsed", width="stretch")

                            else :
                                st.markdown(":small[Récupération des coordonnées...]")
                                st.session_state.clicked = tuple(st.session_state.clicked.values())
                                st.session_state.last_clicked = st.session_state.clicked
                                st.rerun()


                        case "Autre (ajouter une remarque)":
                            if st.session_state.clicked is not None or st.session_state.last_clicked is not None:
                                st.session_state.clicked = None
                                st.session_state.last_clicked = None
                                st.rerun()
                            st.session_state.type_annotation = "remarque"
                            select_key = f"select_remarque_{st.session_state.id_obs}"
                            st.text_area("Saisissez ici vos remarques", key=select_key)

                        case None:
                            if st.session_state.clicked is not None or st.session_state.last_clicked is not None:
                                st.session_state.clicked = None
                                st.session_state.last_clicked = None
                                st.rerun()
                            st.session_state.type_annotation = None
                    
                    # TODO: ajouter une option d'annotation milieu/micromilieu, en faisant appel à la liste Villaret. 
                    # La proposition des milieux peut être faite en fonction de l'espèce considérée (selon qu'elle est présente dans la liste d'espèce du dictionnaire ou non)

                    # TODO: ajouter une option de modification de la position, en utilisant un pointage sur la carte interactive

                    # validation radio should also be unique per observation so it resets on id change
                    st.radio("Validation de la donnée:", ['Je confirme', 'Donnée douteuse', "Donnée fausse"], key=validation_key)

                    # pass the validation widget key so the callback reads the current value at execution time
                    st.form_submit_button("Sauvegarder l'annotation", on_click=_save_annotation, args=(id_obs, validation_key, st.session_state.type_annotation))


        ###################################################
        # Affichage des metadonnees
        with col_meta:
            st.subheader("Metadonnées de l'observation")
            
            if check_filtered_and_clicked(st.session_state.filtered, st.session_state.id_obs) : 
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, filtered_data, st.session_state.id_obs, st.session_state.last, st.session_state.type_annotation)
                afficher_metadonnees(filtered_data, 
                            st.session_state.id_obs, 
                            st.session_state.output_data)
        
        st.subheader("Annotations de la session en cours")
        if len(st.session_state.output_data)!=0:
            st.dataframe(st.session_state.output_data,
                         hide_index=True,
                         column_order=("ID", species_column, "Date_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation", "Atypicité_NFaure", "Atypicité_Kohonen", "Atypicité_Fréquence", ))
        else:
            st.write("Les annotations s'afficheront ici une fois enregistrées.")

    ###################################################
    # DEUXIEME ONGLET POUR LES STATISTIQUES
    with tab2: 
        if check_filtered_and_clicked(st.session_state.filtered, st.session_state.id_obs) :
            
            mask = data['ID'] == id_obs
            if not mask.any():
                st.error(f"ID inconnu dans les données filtrées : {id_obs}")
            else :
                row = data.loc[mask].iloc[0]

            col_geo, col_atyp, col_rel = st.columns([1, 1, 1], border= True, gap=None)
            
            id_obs = st.session_state.id_obs
            
            ###################################################
            # Colonne pour les données spatiales
            with col_geo :
                st.subheader("Données spatiales")
                afficher_stats_geo(filtered_data, 
                            id_obs, 
                            st.session_state.output_data)
                

                st.text(f"Carte des observations de {row[species_column]} :",
                          help = "L'ensemble des observations disponibles pour l'espèce sont affichées. La couleur des points correspond à leur atypicité. Une atypicité non calculable avec la méthode sélectionnée correspond à la couleur blanche. Il n'est pas possible de sélectionner une observation depuis cet affichage.")
                
                map2, group2 = make_map(data.loc[data[species_column]==row[species_column]],
                                colormap,
                                annotated= st.session_state.output_data['ID'].to_list(),
                                toggle_clusters=False, 
                                toggle_dpt=st.session_state.dpt)

                st_data2 = st_folium(map2, key='map2', 
                                     width=__width__, height=__height__)
                    
            ###################################################
            # Colonne pour les données de fréquence et d'atypicité       
            with col_atyp :
                st.subheader("Données de fréquence")
                afficher_stats_atypicite(data, 
                            id_obs, 
                            st.session_state.output_data,
                            st.session_state.filters['Méthode'])
                
                proportion_lower_atypicity = metrics.compute_proportion_lower_atypicity(data, 
                                                                                species_column,
                                                                                species = row[species_column], 
                                                                                atypicity = row[st.session_state.filters["Méthode"]],
                                                                                atypicity_column = st.session_state.filters["Méthode"])

                st.text(f"-> cette observation est plus atypique que {proportion_lower_atypicity}% des observations de l'espèce", 
                        help="Attention, les données dont l'atypicité ne peut pas être calculée avec la méthode choisie sont ignorées.")
                
                ###################################################
                # Histogramme des atypicités de l'espèce
                st.text(f"Histogramme des atypicités des spécimens de {row[species_column]} :",
                        help="La ligne rouge correspond à l'observation en cours d'affichage. Les colonnes bleues correspondent au nombre d'observations pour une valeur d'atypicité donnée")

                fig, ax = plt.subplots()
                ax.hist(data.loc[data[species_column]==row[species_column]][st.session_state.filters["Méthode"]],
                        bins=10, color="blue")
                plt.axvline(x=row[st.session_state.filters["Méthode"]], color="red", label=f"Atypicité(observation) = {round(row[st.session_state.filters['Méthode']], 3)}")
                #TODO : fix first loading
                
                plt.xlim((0,10))
                plt.xlabel("Atypicité")
                plt.ylabel("Occurences")
                ax.legend()
                st.pyplot(fig)
                
                st.write("")
                
                ###################################################
                # Boxplot des atypicités intra- et inter-relevés
                st.text("Comparaison des atypicités intra- et inter-relevés :",
                         help="Boxplot. L'atypicité moyenne de chaque relevé a été calculée pour obtenir le profil de droite. En rouge, les médianes.")
                
                profil_intra, profil_inter = get_intra_inter_atypicity(data, Code_Releve=row["Code_Releve"], method=st.session_state.filters['Méthode'])
                
                fig, ax = plt.subplots()
                ax.boxplot([profil_intra, profil_inter])
                ax.set_xticklabels([f"Profil du relevé n°{row['Code_Releve']} \n (Intra)", 
                                    "Profil de l'ensemble des relevés \n (Inter)"])
                plt.ylabel("Atypicité")
                st.pyplot(fig)
                
                
            ###################################################
            # Colonne pour les données relatives au relevé et au milieu 
            with col_rel :
                st.subheader("Données du relevé")
                
                afficher_stats_releve(data, id_obs)
                
                # dataframe et sets contenant les infos du relevé
                df_releve, set_releve = get_releve(data, row['Code_Releve'])
                set_milieu = get_set_milieu(set_releve, milieu_pour_chaque_espece)
                
                ###################################################
                # Tableau des observations
                st.write("**Observations du relevé** :")
                st.dataframe(df_releve, 
                             hide_index=True, 
                             column_order=("ID", species_column, "Nom_Valide", "NbObs", st.session_state.filters["Méthode"]),
                             height = min(200, len(df_releve)*60))
                

                st.write("")
                
                ###################################################
                # Tableau des indices de Jaccard
                df_Jaccard = get_df_Jaccard(set_releve, set_milieu, especes_pour_chaque_milieu, dict_milieux)
                
                if len(df_Jaccard)!=0:
                    st.text("Milieux probables (indice de Jaccard) :",
                            help="L'indice de Jaccard mesure la similitude entre 2 ensembles d'espèces. Ici, on compare les espèces du relevé à la littérature (Villaret). Une valeur de 1 équivaut à une correspondance parfaite, et une valeur de 0 à une absence de correspondance.") 
                    st.dataframe(df_Jaccard, height=200)
                else:
                    
                    st.text("Milieux probables :",
                            help="Les données utilisées, issues de la littérature (Villaret), ne contiennent pas toutes les espèces présentes en Isère.") 
                    st.markdown("-> observations insuffisantes pour calculer l'indice de Jaccard", unsafe_allow_html=True)
                
                st.write("")
                
                ###################################################
                # Boxplot des indices d'Ellenberg
                use_cols_Ellenberg = ['Lumiere', 'Temperature', "Humidite_edaphique", "Reaction_du_sol_(pH)", "Matiere_organique", "Texture"]
                df_fact_abiotiques = get_df_fact(data_fact_abiotiques, df_releve, 
                                                 cols=use_cols_Ellenberg, species_column=species_column)
                
                if len(df_fact_abiotiques)!=0:
                    st.text("Profil écologique :",
                            help = "Les indices d'Ellenberg mesurent les préférences d'une espèce pour différents facteurs abiotiques. Synthétiser les indices d'Ellenberg du relevé peut permettre de comprendre le type de milieu qu'il représente.")
                    
                    fig, ax = plt.subplots()
                    ax.boxplot(df_fact_abiotiques[use_cols_Ellenberg])
                    ax.grid(axis="both", color="lightgrey", linewidth=0.7)
                    plt.xticks([i for i in range(1,len(use_cols_Ellenberg)+1)], 
                               labels=[i.replace("_", "\n", 1) for i in use_cols_Ellenberg], 
                               rotation=60)
                    st.pyplot(fig)
                    
                else :
                    st.text("Profil écologique :",
                            help = "Les données utilisées ne contiennent pas toutes les espèces présentes en Isère. Aucune information n'a été trouvée concernant les préférences écologiques des espèces du relevé")
                    st.markdown("-> observations insuffisantes pour calculer le profil écologique", unsafe_allow_html=True)
                
                
            ###################################################
            # Affichage d'un tableau de donnees supplementaire
    
            st.subheader(f"Données brutes (n = {len(filtered_data)})")
            st.dataframe(filtered_data.head(100), 
                        hide_index=True,
                        column_order=("ID", species_column, "Nom_Valide", "Latitude", "Longitude", "PrenomNom", "NbObs", "Frequence", "Groupe", "Atypicité_NFaure", "Atypicité_Kohonen", "Atypicité_Fréquence", "Atypicité_Hybride", "rank_ground_truth", "RangEspUC", "Code_Releve", "Date_Releve", "NbObs_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation"))
# TODO : ajouter code postal/commune
# TODO : superposition cartes
                
