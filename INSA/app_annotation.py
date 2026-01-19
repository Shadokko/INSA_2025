from pydoc import doc
import streamlit as st
import folium
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
params = yaml.load(path2param)
DATA_PATH_NFAURE = params['DATA_PATH_NFAURE']
print(f"N. Faure's data path: {Path(DATA_PATH_NFAURE).resolve()}\n")

DATA_PATH_KOHONEN = params['DATA_PATH_KOHONEN']
print(f"Kohonen's data path: {Path(DATA_PATH_KOHONEN).resolve()}\n")

species_list_path = params['species_list_path']
print(f"Species list path: {Path(species_list_path).resolve()}\n")

export_path = Path(params['export_path'])
print(f"Export path: {export_path.resolve()}\n")

ISERE = params['ISERE']
__width__ = params['width']
species_column = params['species_column']


# DATA_PATH_NFAURE = "../../result_export.csv"
# DATA_PATH_KOHONEN = "experiments/kohonen_results.csv"


__GRENOBLE__ = (45.0106, 9.4330)
colormap = cm.LinearColormap(["green", "yellow", "red", "purple"], vmin=0, vmax=10, caption="échelle d'atypicité")


# TODO: ajouter l'inférence et la détermination de rank_ground_truth, en faisant appel à un modèle pré-entraîné
# TODO: ajouter la "Note" d'Alain comme option, ainsi que la fréquence, et proposer un indicateur synthétique. Une classe atypicité pourra être créée.
@st.cache_data
def compute_atypicity(filtered_data, data, method):
    """
    Calcule un score d'atypicité normalisé sur l'échelle 0-10, et ajoute une colonne "Atypicité" correspondante

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
        Méthode de calcul : "rank_ground_truth", "kohonen", ou "frequency".

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

    match method:
        case "rank_ground_truth":
            minv = np.min(data["rank_ground_truth"])
            maxv = np.max(data["rank_ground_truth"])
            print(f"Rank_ground_truth min: {minv}, max: {maxv}")
            denom = maxv - minv
            if denom != 0:
                return 10 * (filtered_data["rank_ground_truth"].values - minv) / denom
            else:
                return np.zeros(len(filtered_data))

        case "kohonen":
            minv = np.min(data["RangEspUC"])
            maxv = np.max(data["RangEspUC"])
            print(f"Kohonen RangEspUC min: {minv}, max: {maxv}")
            denom = maxv - minv
            if denom != 0:
                return 10 * (filtered_data["RangEspUC"].values - minv) / denom
            else:
                return np.zeros(len(filtered_data))
        
        case "frequency":
            # Return an atypicity score based on species frequency in the whole dataset
            species_counts = data[species_column].value_counts(normalize=True)
            return filtered_data[species_column].map(lambda x: 10 * (1 - species_counts.get(x, 0))).values
        
        case "hybrid":
            st.error("Méthode 'hybrid' non implémentée.")
            return np.zeros(len(filtered_data))

# TODO: charger aussi une liste de milieux "villaret", à des fins d'annotations milieux
# TODO: charger les annotations déjà effectuée en initialisant st.session_state.output_data
@st.cache_data
def load_data(path_nfaure, path_kohonen):
    """
    Charge les données 

    Parameters
    ----------
    path_nfaure : string
        chemin vers les données du modèle de N. Faure
    path_kohonen : string
        chemin vers les données du modèle de Kohonen

    Returns
    -------
    data : pandas data frame
        tableau contenant uniquement les données utiles
        
    observateurs : list
        liste des observateurs présents dans les données, sans répétition
    
    especes : list
        liste des espcèces présentes dans les données, sans répétition
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
    data["Atypicité_NFaure"] = compute_atypicity(data, data, "rank_ground_truth")
    data["Atypicité_Kohonen"] = compute_atypicity(data, data, "kohonen")
    data["Atypicité_Frequency"] = compute_atypicity(data, data, "frequency")
    # data["Atypicité_Hybrid"] = compute_atypicity(data, data, "hybrid")
    
    # Par défaut pour la carte, on peut créer une colonne 'Atypicité' basée sur le filtre actif
    if "filtered_data" in st.session_state and st.session_state.filtered_data is not None:
        method = st.session_state.filters["Méthode"]
        if method == "rank_ground_truth":
            data["Atypicité"] = data["Atypicité_NFaure"]
        elif method == "kohonen":
             data["Atypicité"] = data["Atypicité_Kohonen"]
        elif method == "frequency":
             data["Atypicité"] = data["Atypicité_Frequency"]
        else:
            st.error(f"Méthode '{method}' non implémentée pour la colonne 'Atypicité'.")
    else:
        data["Atypicité"] = data["Atypicité_Frequency"] # valeur par défaut

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
        especes = list(data[species_column].unique())

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
def filter_data(data, filters):
    """
    Charge les données dans un data frame

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
    
    if (len(st.session_state.filters["PrenomNom"]) == 0) and (len(st.session_state.filters[species_column]) == 0): # on vérifie que l'utilisateur a choisi au moins un observateur ou une espèce, sinon il y a trop de données à afficher, et c'est plus logique
        st.error("Veuillez choisir un observateur ou une espèce.") # erreur qui s'affiche si ce n'est pas le cas
    else :
        if len(st.session_state.filters["PrenomNom"]) != 0: # si il y a un filtre sur l'observateur
            filtered_data = filtered_data.loc[[element in filters["PrenomNom"] for element in filtered_data["PrenomNom"]]] # on filtre par rapport à l'observateur
        if len(st.session_state.filters[species_column]) != 0: # si il y a un filtre sur l'espèce
            filtered_data = filtered_data.loc[[element in filters[species_column] for element in filtered_data[species_column]]] # on filtre par rapport à l'espèce
        if st.session_state.filters["Debut"] > st.session_state.filters["Fin"] : # si le dates choisies ne sont pas dans le bon ordre
            st.error("Veuillez choisir une date de début antérieure à la date de fin.") # erreur affichée
        else :
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date >= filters["Debut"]] # sinon, on garde uniquement les données ultérieures à la date de début choisie
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date <= filters["Fin"]] # puis, on garde uniquement les données précédant la date de début choisie
        
        filtered_data["Atypicité"] = compute_atypicity(filtered_data, data, st.session_state.filters["Méthode"])
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]<st.session_state.filters["hi_Score"]]
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]>st.session_state.filters["lo_Score"]]
        filtered_data = filtered_data.sort_values(by="Atypicité", ascending=False).head(int(st.session_state.filters['Top_atypicity']))
    return filtered_data


def compute_center(data):
    """
    Calcule le centre géographique des données

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

        folium.CircleMarker(location=list(row.loc[['Latitude', 'Longitude']]),
                            radius=radius,
                            color="black",
                            fill=True,
                            fill_color=colormap(float(row.loc['Atypicité'])),
                            fill_opacity=1,
                            popup=f"ID : {row.ID}<br> espèce : {row[species_column]}<br> atypicité : {round(row.loc['Atypicité'], 3)}", 
                            tooltip=f"ID : {row.ID}<br> espèce : {row[species_column]}<br> atypicité : {round(row.loc['Atypicité'], 3)}"
                            ).add_to(group_1)

    if toggle_clusters : # affichage groupé des observations
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1.add_to(marker_cluster)
    else :
        group_1.add_to(map_)


    return map_, group_1
    # return st_folium(map_, width=width, height=height, key=key, on_change=callback)


def update_id_obs(st_data, current, last):
    """
    Met à jour l'ID de l'observation sélectionnée à partir des données
    d'évènement renvoyées par le composant Folium / Streamlit-Folium.

    La fonction gère plusieurs formats possibles renvoyés par le composant :
    - ``last_object_clicked_popup`` : texte du popup contenant "ID : <n>"
    - ``last_object_clicked`` : dictionnaire avec un champ ``id``
    - ``last_object_clicked`` : dictionnaire avec ``lat`` et ``lng`` — on recherche
      alors le marqueur le plus proche parmi ``st.session_state.filtered_data``.

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
    elif clicked:
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
                    df = st.session_state.filtered_data
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
    else:
        lines.append(":red[**observation en attente d'annotation**]")
        lines.append(f"**Espèce ({species_column}):** {row[species_column]}")


    if pd.notna(row.get('annotation_espece')) and row.get('annotation_espece'):
        lines.append(f":green[**Espèce corrigée :** {row['annotation_espece']}]")
    lines.append(f"**Groupe :** {row['Groupe']}")
    lines.append(f"**Observateur :** {row['PrenomNom']}")
    lines.append(f"**Date :** {pd.to_datetime(row['Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")
    lines.append(f"**Coordonnées :** ({row['Latitude']}, {row['Longitude']})")
    if pd.notna(row.get('annotation_latitude')) or pd.notna(row.get('annotation_longitude')): 
        lines.append(f":green[**Coordonnées corrigées :** {row.get('annotation_remarque')}]")
    if pd.notna(row.get('NbObs')):
        lines.append(f"**Spécimens observés :** {int(row['NbObs'])}")
    # Affichage des scores d'atypicité comparés
    score_nf = round(row['Atypicité_NFaure'], 3) if pd.notna(row.get('Atypicité_NFaure')) else np.nan
    score_ko = round(row['Atypicité_Kohonen'], 3) if pd.notna(row.get('Atypicité_Kohonen')) else np.nan
    score_freq = round(row['Atypicité_Frequency'], 3) if pd.notna(row.get('Atypicité_Frequency')) else np.nan
    lines.append(f"**Scores d'atypicité :**")
    lines.append(f"- N. Faure : {f'{score_nf} / 10' if not pd.isna(score_nf) else 'N/A'}")
    lines.append(f"- Kohonen : {f'{score_ko} / 10' if not pd.isna(score_ko) else 'N/A'}")
    lines.append(f"- Basé sur la fréquence : {f'{score_freq} / 10' if not pd.isna(score_freq) else 'N/A'}")
    if pd.notna(row.get('annotation_remarque')):
        lines.append(f":green[**Remarque :** {row['annotation_remarque']}]")

    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

@st.cache_data
def afficher_metadonnees_releve(data, id_obs):
    """
    Affiche des informations complémentaires sur le relevé auquel
    appartient l'observation sélectionnée (code, date, nombre d'observations).

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

    st.write("------")
    st.write(f"code relevé : {row['Code_Releve']}")
    st.write(f"date relevé : {row['Date_Releve']}")
    st.write(f"observations dans le relevé : {row['NbObs_Releve']}")


actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Signaler un micro-milieux"]

def annoter(data, action, id_obs, especes):
    """
    Effectue une action d'annotation sur une observation donnée.

    Cette fonction modifie la table d'annotations en mémoire
    (``st.session_state.output_data``) en fonction de l'action choisie.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame contenant les observations (non modifié directement ici).
    action : str
        Action choisie (une des valeurs de ``actions_possibles``).
    id_obs : int
        Identifiant de l'observation ciblée.
    especes : list
        Liste des espèces possibles (utilisée pour la sélection).
    """
    if action is None:
        return

    match action:
        case "Modifier l'espèce/le nom de l'espèce":
            st.session_state.output_data.at[id_obs, "annotation_espece"] = st.selectbox("Nom de l'espèce", especes)
        case "Signaler un micro-milieux":
            st.session_state.output_data.at[id_obs, "annotation_micro"] = st.text_area("Description", "")
    # Autres actions possibles (position, validation, remarque) sont commentées
    # et prêtes à être implémentées si nécessaire.

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

def _save_annotation(id_obs, validation_key):
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
    if id_obs is None:
        st.error("Aucune observation sélectionnée — impossible de sauvegarder.")
        return

    if 'output_data' not in st.session_state:
        st.session_state.output_data = pd.DataFrame(columns=data.columns)

    # read the current validation status from session state using the provided key
    validation_status = st.session_state.get(validation_key, None)

    # Get the currently selected annotated species for this observation (if any)
    select_key = f"select_espece_{id_obs}"
    annotation_espece = st.session_state.get(select_key, None)

    row = data.loc[data["ID"]==id_obs].copy()
    # ensure annotation_espece column exists
    if 'annotation_espece' not in row.columns:
        row['annotation_espece'] = None
    if annotation_espece is not None:
        row.loc[:, 'annotation_espece'] = annotation_espece

    row = row.assign(validation=validation_status)

    st.session_state.output_data = pd.concat([
        st.session_state.output_data,
        row
    ], ignore_index=True).drop_duplicates(subset=['ID'], keep='last')

    st.success(f"Annotation sauvegardée pour l'observation ID {id_obs}.")

if __name__ == "__main__":

    tab1, tab2 = st.tabs(["Visu&Annotation", "Statistiques"])

    ###################################################
    # Chargement des donnees
    # msg = st.toast("Chargement des données...")
    data, observateurs, especes = load_data(DATA_PATH_NFAURE, DATA_PATH_KOHONEN)

    if "output_data" not in st.session_state:
        st.session_state.output_data =  pd.DataFrame(columns=data.columns)


    # msg.toast("Données à jour !", icon=":material/check:")
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = __GRENOBLE__
        st.session_state["map_zoom"] = 8

    if "filtered" not in st.session_state:
        st.session_state.filtered = False
        st.session_state.filtered_data = None
        st.session_state.id_obs = None
        st.session_state.last = None
        st.session_state.afficher_releve = False
        st.session_state.afficher_espece = False

    ###################################################
    # Premier onglet pour la visualisation

    with tab1:
        col_carte, col_annot, col_meta = st.columns([5, 3, 2], border= True, gap=None) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees

        ###################################################
        # Selection des filtres
        
        st.sidebar.subheader("Filtres") # menu de selection des filtres
        with st.sidebar.form(key="filtres"):
            st.session_state.filters = dict()
            st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", sorted(observateurs), placeholder="Aucune sélection")
            st.session_state.filters[species_column] = st.multiselect("Espèce", sorted(especes), placeholder="Aucune sélection")
            st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["lo_Score"], st.session_state.filters["hi_Score"] = st.select_slider("Atypicité", options=[i for i in np.arange(0, 10.5, 0.5)], value=(0,10))
            st.session_state.filters['Top_atypicity'] = st.slider('Filter les plus atypiques', min_value=5, max_value=100, value=20, step=5)
            # st.markdown('''0 :green[----------]:yellow[----------]:orange[----------]:red[----------]:violet[----------] 10''') # légende
            st.session_state.filters["Méthode"] = st.radio("Méthode de calcul de l'atypicité :", ["rank_ground_truth", "kohonen", "frequency"])

            st.session_state.filtered = st.form_submit_button(label="Enregistrer") # validation des filtres
            
            if st.session_state.filtered:
                with st.sidebar.status("Sélection des données et calcul...") as status:
                    # 1. Filtrage de base (dates, observateurs, etc.)
                    st.session_state.filtered_data = filter_data(data, st.session_state.filters)

                    # 2. Mise à jour de la colonne 'Atypicité' selon la méthode choisie
                    st.session_state.filtered_data["Atypicité"] = compute_atypicity(st.session_state.filtered_data, 
                                                                                    data, st.session_state.filters["Méthode"])
                    
                    # 3. Application du filtre "Top atypicity" sur la base du score choisi
                    st.session_state.filtered_data = st.session_state.filtered_data.sort_values(
                        by="Atypicité", ascending=False
                    ).head(int(st.session_state.filters['Top_atypicity']))
                    
                    status.update(label='Données prêtes', state="complete")

        clusters = st.sidebar.toggle("Affichage groupé")
        st.session_state.dpt = st.sidebar.toggle("Afficher le département de l'Isère", 
                                                value=True)


        ###################################################
        # Affichage de la carte
        with col_carte:
            sub_col_carte_1, sub_col_carte_2 = st.columns(2)
            
            sub_col_carte_1.subheader("Carte des observations")
            
            if type(st.session_state.filtered_data) == type(None):
                st.write("Veuillez filtrer les données")
            elif len(st.session_state.filtered_data) == 0:
                st.error("Aucune observation ne correspond à ces critères. Veuillez essayer avec une autre méthode de calcul de l'atypicité ou en élargissant les filtres.")
            else :
                map1, group1 = make_map(st.session_state.filtered_data,
                                colormap,
                                annotated= st.session_state.output_data['ID'].to_list(),
                                toggle_clusters=clusters, 
                                toggle_dpt=st.session_state.dpt)

                st_data1 = st_folium(map1, key='map1', 
                                     width=__width__)
        
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, st.session_state.id_obs, st.session_state.last)


        #########################
        # Formulaire d'annotation
        with col_annot:

            st.subheader("Formulaire d'annotation")    
            if type(st.session_state.filtered_data) == type(None):
                st.write("Veuillez filtrer les données")
            elif st.session_state.id_obs is None: # si aucune observation n'a ete selectionnee
                st.write("Veuillez cliquer sur une observation pour l'annoter")
            else: 
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, st.session_state.id_obs, st.session_state.last)
                st.button("Exporter les annotations", on_click=lambda: save_annotations(st.session_state.output_data, export_path))
     
                form_key = f"annotation_{st.session_state.id_obs}"
                select_key = f"select_espece_{st.session_state.id_obs}"
                validation_key = f"validation_{st.session_state.id_obs}"

                with st.form(key=form_key):
                    st.subheader("Annotation de l'observation")
                    # Update the selected id from the map component early so it is preserved across reruns

                    id_obs = st.session_state.id_obs

                    # Prefer previously saved annotation (in output_data) if present, otherwise use recorded species
                    default_espece = None
                    if hasattr(st.session_state, "output_data") and not st.session_state.output_data.empty:
                        prev = st.session_state.output_data.loc[st.session_state.output_data['ID']==id_obs, 'annotation_espece']
                        if not prev.empty and pd.notna(prev.iloc[-1]):
                            default_espece = prev.iloc[-1]

                    if default_espece is None and (st.session_state.filtered_data is not None) and (id_obs in st.session_state.filtered_data.index):
                        if st.session_state.filtered_data.at[id_obs, 'annotation_espece'] and pd.notna(st.session_state.filtered_data.at[id_obs, 'annotation_espece']):
                            default_espece = st.session_state.filtered_data.at[id_obs, 'annotation_espece']
                        else:
                            default_espece = st.session_state.filtered_data.at[id_obs, species_column] if id_obs in st.session_state.filtered_data.index else None

                    if default_espece in especes:
                        default_index = especes.index(default_espece)
                    else:
                        default_index = 0

                    st.selectbox(f"Modifier l'espèce (Valeur initiale: {default_espece})", especes, index=default_index, key=select_key)
                    
                    # TODO: ajouter une option d'annotation milieu/micromilieu, en faisant appel à la liste Villaret. 
                    # La proposition des milieux peut être faite en fonction de l'espèce considérée (selon qu'elle est présente dans la liste d'espèce du dictionnaire ou non)

                    # TODO: ajouter une option de modification de la position, en utilisant un pointage sur la carte interactive

                    # validation radio should also be unique per observation so it resets on id change
                    st.radio("Validation de la donnée:", ['Je confirme', 'Donnée douteuse', "Donnée fausse"], key=validation_key)

                    # pass the validation widget key so the callback reads the current value at execution time
                    st.form_submit_button("Sauvegarder l'annotation", on_click=_save_annotation, args=(id_obs, validation_key))


        ###################################################
        # Affichage des metadonnees
        with col_meta:
            st.subheader("Metadonnées de l'observation")
            if type(st.session_state.filtered_data) == type(None):
                st.write("Veuillez filtrer les données")
            elif st.session_state.id_obs is None: # si aucune observation n'a ete selectionnee
                st.write("Veuillez cliquer sur une observation pour afficher ses métadonnées")
            else: 
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, st.session_state.id_obs, st.session_state.last)
                afficher_metadonnees(st.session_state.filtered_data, 
                            st.session_state.id_obs, 
                            st.session_state.output_data)


    ###################################################
    # DEUXIEME ONGLET POUR LES STATISTIQUES
    with tab2: 
        # TODO: add statistics on number of obs, number of filtered obs, number of species, histogram on Atypicity

        ###################################################
        # Affichage d'un tableau de donnees supplementaire
   
        if type(st.session_state.filtered_data) != type(None):
            st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
            select_row = st.dataframe(st.session_state.filtered_data.head(100), 
                        hide_index=True, 
                        selection_mode="single-row",
                        on_select="rerun",
                        column_order=("ID", species_column, "Nom_Valide", "Latitude", "Longitude", "PrenomNom", "NbObs", "Groupe", "Atypicité_NFaure", "Atypicité_Kohonen", "Atypicité_frequency", "rank_ground_truth", "RangEspUC", "Code_Releve", "Date_Releve", "NbObs_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation"))
                
        else :
            st.subheader("Données brutes")
            st.write("Veuillez filtrer les données")

    ###################################################
    # Deuxième onglet pour l'annotation et l'affichage de données supplémentaires

    # skip this part

    # with tab2:
    #     if type(st.session_state.filtered_data) == type(None):
    #         st.write("Veuillez filtrer les données")
    #     elif type(st_data1['last_object_clicked']) == type(None): # si aucune observation n'a ete selectionnee
    #         st.write("Veuillez cliquer sur une observation pour afficher les données associées")
    #     else : 
    #         col_carte, col_data = st.columns([2, 2], border= True) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees

    #         with col_carte:
    #             row1 = st.container(height=475)
    #             row2 = st.container(height=475)
                
    #             ###################################################
    #             # Affichage d'une carte centrée et zoomée sur l'observation
                
    #             with row1:
    #                 sub_col_carte_1, sub_col_carte_2 = st.columns(2)
                    
    #                 sub_col_carte_1.subheader("Cartes")
    #                 afficher_espece = sub_col_carte_2.toggle("Afficher l'espèce")

    #                 if afficher_espece :
    #                     st_data2 = make_map(data.loc[data[species_column]==data.at[st.session_state.id_obs, species_column]],
    #                             colormap, 
    #                             360,
    #                             360,
    #                             100,
    #                             clusters,
    #                             key=2, 
    #                             center=(st_data1['last_object_clicked']["lat"], st_data1['last_object_clicked']["lng"]))
    #                 else :
    #                     st_data2 = make_map(data.iloc[[st.session_state.id_obs]], 
    #                             colormap,
    #                             360,
    #                             360,
    #                             1,
    #                             clusters,
    #                             key=2, 
    #                             center=(st_data1['last_object_clicked']["lat"], st_data1['last_object_clicked']["lng"]))
                    
    #             ###################################################
    #             # Annotation
                
    #             with row2 :
    #                 st.subheader("Annotation")
    #                 actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Modifier la position", "Signaler un micro-milieux", "Valider l'observation", "Autre"]
    #                 action = st.selectbox("Que souhaitez-vous faire ?", actions_possibles, index=None, placeholder="Veuillez choisir une option")
    #                 annoter(data, action, st.session_state.id_obs, especes)
                    
    #         ###################################################
    #         # Afficahge de données supplémentaires

    #         with col_data:
    #             sub_col_data_1, sub_col_data_2 = st.columns(2)
                
    #             sub_col_data_1.subheader("Metadonnées")
    #             afficher_releve = sub_col_data_2.toggle("Afficher le relevé")

    #             afficher_metadonnees(data, st.session_state.id_obs)
    #             if afficher_releve :
    #                 afficher_metadonnees_releve(data, st.session_state.id_obs)
    #                 st.dataframe(data.loc[
    #                     data["Code_Releve"]==data.at[st.session_state.id_obs, 'Code_Releve']
    #                     ], hide_index=True, column_order=("ID", species_column, "Nom_Valide", "NbObs", "Atypicité"))
                    
    #             # TODO : ajouter code postal/commune, nbr de communes où l'espèce est présente
    #             # TODO : superposition cartes
                
