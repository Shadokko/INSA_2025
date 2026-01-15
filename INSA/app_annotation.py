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
import matplotlib.pyplot as plt

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
DATA_PATH = params['DATA_PATH']
print(f"Data path: {Path(DATA_PATH).resolve()}\n")

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


# DATA_PATH = "../../result_export.csv"


__GRENOBLE__ = (45.0106, 9.4330)
colormap = cm.LinearColormap(["green", "yellow", "red", "purple"], vmin=0, vmax=10, caption="échelle d'atypicité")


# TODO: ajouter l'inférence et la détermination de rank_ground_truth, en faisant appel à un modèle pré-entraîné
# TODO: ajouter la "Note" d'Alain comme option, ainsi que la fréquence, et proposer un indicateur synthétique. Une classe atypicité pourra être créée.
@st.cache_data
def compute_atypicity(filtered_data, data, method):
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
        case "rank_ground_truth":
            minv = np.min(data["rank_ground_truth"])
            maxv = np.max(data["rank_ground_truth"])
            denom = maxv - minv
            if denom == 0:
                # Si toutes les valeurs sont identiques, on renvoie des zéros
                return np.zeros(len(filtered_data))
            return 10 * (filtered_data["rank_ground_truth"] - minv) / denom


# TODO: charger les annotations déjà effectuée en initialisant st.session_state.output_data
@st.cache_data
def load_data(filename):
    """
    Charge les données 

    Parameters
    ----------
    filename : string
        chemin vers les données
        
    Returns
    -------
    data : pandas data frame
        tableau contenant uniquement les données utiles
        
    observateurs : list
        liste des observateurs présents dans les données, sans répétition
    
    especes : list
        liste des espcèces présentes dans les données, sans répétition
    """
    chunk_size = 10_000 # les données seront chargées par paquets pour aller plus vite
    chunks = [] # liste qui contiendra tous les paquets de données

    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", species_column, "NbObs", "Nom_Valide", "Groupe", "Date_Releve", "Code_Releve", "NbObs_Releve", "Frequence_espece"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data["ID"] = data.index # on rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique
    data["Atypicité"] = compute_atypicity(data, data, "rank_ground_truth")
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
def load_Villaret(filename):
    """
    Importe les données isues de l'ouvrage de Villaret

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
    chunk_size = 1_000 # les données seront chargées par paquets pour aller plus vite
    chunks = [] # liste qui contiendra tous les paquets de données

    for chunk in pd.read_csv(filename, sep=",", header=1,chunksize=chunk_size, 
                             usecols=["Nom flore", "Lumiere", "Temperature", "Humidite_edaphique", "Reaction_du_sol_(pH)", "Niveau_trophique", "Matiere_organique", "Salinite", "Texture", "Continentalite"]):
        chunks.append(chunk)
    data_fact = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data_fact = data_fact.dropna()
    
    return data_fact

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
    
    if (len(st.session_state.filters["PrenomNom"]) == 0) and (len(st.session_state.filters[species_column]) == 0) and not st.session_state.filters["ID"]: # on vérifie que l'utilisateur a choisi au moins un observateur ou une espèce, sinon il y a trop de données à afficher, et c'est plus logique
        st.error("Veuillez choisir un observateur, une espèce ou un ID.") # erreur qui s'affiche si ce n'est pas le cas
    else :
        if len(st.session_state.filters["PrenomNom"]) != 0: # si il y a un filtre sur l'observateur
            filtered_data = filtered_data.loc[[element in filters["PrenomNom"] for element in filtered_data["PrenomNom"]]] # on filtre par rapport à l'observateur
        if len(st.session_state.filters[species_column]) != 0: # si il y a un filtre sur l'espèce
            filtered_data = filtered_data.loc[[element in filters[species_column] for element in filtered_data[species_column]]] # on filtre par rapport à l'espèce
        if st.session_state.filters["ID"]:
            filtered_data = filtered_data.loc[filtered_data["ID"]==st.session_state.filters["ID"]]
        if st.session_state.filters["Debut"] > st.session_state.filters["Fin"] : # si le dates choisies ne sont pas dans le bon ordre
            st.error("Veuillez choisir une date de début antérieure à la date de fin.") # erreur affichée
        else :
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date >= filters["Debut"]] # sinon, on garde uniquement les données ultérieures à la date de début choisie
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date <= filters["Fin"]] # puis, on garde uniquement les données précédant la date de début choisie
        
        # filtered_data["Atypicité"] = compute_atypicity(filtered_data, data, st.session_state.filters["Méthode"])
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
        if int(row['ID']) in annotated: radius = 3
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
    current : int | None
        ID actuellement sélectionné.
    last : int | None
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
        m = re.search(r'ID\s*:\s*(\d+)', popup)
        if m:
            try:
                new = int(m.group(1))
            except:
                new = None
    elif clicked:
        if isinstance(clicked, dict):
            if 'id' in clicked:
                try:
                    new = int(clicked['id'])
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
                        new = int(df.at[nearest_idx, 'ID'])
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
def date_francaise(date):
    date_anglais = pd.to_datetime(date,format='%Y-%m-%d').strftime('%d %B %Y')
    
    correspondance = {"January" : "Janvier", "February" : "Février", "March" : "Mars",
                "April" : "Avril", "May":"Mai", "June":"Juin", "July":"Juillet",
                "August":"Août", "September":"Septembre", "October":"Octobre",
                "November":"Novembre", "December":"Décembre"}
    
    for mois_anglais in correspondance.keys():
        if mois_anglais in date_anglais:
            mois_francais = correspondance[mois_anglais]
            return date_anglais.replace(mois_anglais, mois_francais)


@st.cache_data
def afficher_metadonnees(data, id_obs, output_data):
    """
    Affiche, via Streamlit, les métadonnées et le statut d'annotation d'une
    observation sélectionnée.

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
    
    # index_in_filtered_data = int(data["ID"].to_list().index(id_obs))
    lines = []
    lines.append(f"ID : {id_obs}")

    if id_obs in output_data['ID'].to_list():
        lines.append(":green[observation annotée]")

        if output_data.loc[output_data['ID']==id_obs, 'validation'].values[0] == "Je confirme": color = "green"
        elif output_data.loc[output_data['ID']==id_obs, 'validation'].values[0] == "Donnée douteuse": color = "orange"
        else: color = "red"
 
        lines.append(f"Statut: :{color}[{output_data.loc[output_data['ID']==id_obs, 'validation'].values[0]}]")

        annotations_espece = output_data.loc[output_data['ID']==id_obs, 'annotation_espece'].values        
        # if len(annotations_espece != 1):
        #     st.error("Erreur : annotation espèce non unique pour cette observation.")
        annotation_espece = annotations_espece[0]

        if data.at[id_obs, species_column] != annotation_espece:
            lines.append(f"Espèce ({species_column}): {data.at[id_obs, species_column]} :green[→ {annotation_espece}]")
        else:
            lines.append(f"Espèce ({species_column}): {data.at[id_obs, species_column]} :green[espèce non modifiée]")
    else:
        lines.append(":red[observation en attente d'annotation]")
        lines.append(f"Espèce ({species_column}): {data.at[id_obs, species_column]}")


    # lines.append(f"nom valide : {data.at[id_obs, 'Nom_Valide']}")
    if data.at[id_obs, 'annotation_espece']:
        lines.append(f":green[espèce corrigée : {data.at[id_obs, 'annotation_espece']}]")
    lines.append(f"groupe : {data.at[id_obs, 'Groupe']}")
    lines.append(f"observateur : {data.at[id_obs, 'PrenomNom']}")
    lines.append(f"date : {date_francaise(data.at[id_obs, 'Date_Releve'])}")
    lines.append(f"coordonnées : ({data.at[id_obs, 'Latitude']}, {data.at[id_obs, 'Longitude']})")
    if data.at[id_obs, 'annotation_latitude'] or data.at[id_obs, 'annotation_longitude']: 
        lines.append(f":green[coordonnées corrigées : {data.at[id_obs, 'annotation_remarque']}]")
    lines.append(f"spécimens observés : {int(data.at[id_obs, 'NbObs'])}")
    lines.append(f"atypicité : {round(data.at[id_obs, 'Atypicité'], 3)}")
    if data.at[id_obs, 'annotation_remarque']:
        lines.append(f":green[remarque : {data.at[id_obs, 'annotation_remarque']}]")

    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)
    
@st.cache_data
def afficher_stats_geo(data, id_obs, output_data):
    """
    Affiche, via Streamlit, les métadonnées et le statut d'annotation d'une
    observation sélectionnée.

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
    
    # index_in_filtered_data = int(data["ID"].to_list().index(id_obs))
    lines = []
    lines.append(f"**ID** : {id_obs}")
    lines.append(f"**coordonnées** : ({data.at[id_obs, 'Latitude']}, {data.at[id_obs, 'Longitude']})")
    if data.at[id_obs, 'annotation_latitude'] or data.at[id_obs, 'annotation_longitude']: 
        lines.append(f":green[coordonnées corrigées : {data.at[id_obs, 'annotation_remarque']}]")
    lines.append(f"**commune** : {'À implémenter'}")
    
    # Use a single markdown with <br> to avoid extra vertical spacing between lines
    st.markdown("<br>".join(lines), unsafe_allow_html=True)
    
    
@st.cache_data
def afficher_stats_atypicite(data, id_obs, output_data):
    """
    Affiche, via Streamlit, les métadonnées et le statut d'annotation d'une
    observation sélectionnée.

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
    
    # index_in_filtered_data = int(data["ID"].to_list().index(id_obs))
    lines = []
    lines.append(f"**spécimens observés** : {int(data.at[id_obs, 'NbObs'])}")
    lines.append(f"**fréquence de l'espèce** : {float(data.at[id_obs, 'Frequence_espece'][1:-1:])}")
    lines.append(f"**atypicité** : {round(data.at[id_obs, 'Atypicité'], 3)}")
    d = data.loc[data[species_column]==data.at[id_obs, species_column]]
    s = sum(d["Atypicité"] < d.at[id_obs, "Atypicité"])
    lines.append(f"-> cette observation est plus atypique que {round(100*s/(len(d)-1), 1)}% des observations de l'espèce")
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
    lines = []
    lines.append(f"**observateur** : {data.at[id_obs, 'PrenomNom']}")
    lines.append(f"**code relevé** : {data.at[id_obs, 'Code_Releve']}")
    lines.append(f"**date** : {date_francaise(data.at[id_obs, 'Date_Releve'])}")
    lines.append(f"**observations dans le relevé** : {data.at[id_obs, 'NbObs_Releve']}")
    st.markdown("<br>".join(lines), unsafe_allow_html=True)

actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Signaler un micro-milieux"] #TODO

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

@st.cache_data
def indice_jaccard(set1, set2):
    """
    Calcule l'indice de Jaccard entre 2 sets

    Parameters
    ----------
    set1 : set
        set contenant une liste d'espèces (observées ou caractéristiques)
        
    set2 : set
        set contenant une liste d'espèces (observées ou caractéristiques)
        
    Returns
    -------
    indice : float
        indice de Jaccard
    """
    cardinal_union = len(set1|set2)
    cardinal_inter = len(set1 & set2)
    indice = cardinal_inter/cardinal_union
    return cardinal_inter, indice

@st.cache_data
def format_species(espece):
    """
    Retire les suffixes des noms d'espèces (année de découverte et découvreur)
    Format obtenu : "Genre espèce" ou "Genre espèce subsp. sous-espèce" ou"croisée1 x croisée2"

    Parameters
    ----------
    espece : string
        nom d'une espèce
        
    Returns
    -------
    formatted_esp : string
        nom de l'espèce au bon format
        
    """
    l = espece.split()
    if "subsp." in l:
        return " ".join(l[:4])
    elif "x" in l:
        return " ".join(l[:3])
    else:
        return " ".join(l[:2])

@st.cache_data    
def get_releve(data, ID_releve):
    """
    Crée un set contenant les espèces du relevé

    Parameters
    ----------
    df_releve : pandas DataFrame
        subset des données pour les espèces du relevé
        
    Returns
    -------
    set_releve : set
        set contenant les espèces du relevé
    """
    set_releve = set()
    df_releve = data.loc[data["Code_Releve"]==ID_releve]
    
    for esp in df_releve["Nom_Valide"]:
        esp=format_species(esp)
        set_releve.add(esp)
        
    return df_releve, set_releve

@st.cache_data
def get_set_milieu(set_releve, dict_m_pour_e):
    """
    Crée un set contenant les codes d'identification des milieux associés aux observations du relevé

    Parameters
    ----------
    set_releve : set
        espèces observées dans le relevé
        
    dict_m_pour_e : dict
        dictionnaire associant à une espèce les milieux dans lesquels elle est présente
        
    Returns
    -------
    set_milieu : set
        set contenant les milieux qui donneront un indice de Jaccard non nul
        
    Notes
    -----
    - set_milieu ne contient que les milieux qui ont au moins une observation en commun avec le relevé
        
    """
    set_milieu = set()
    for esp in set_releve:
        if esp in dict_m_pour_e.keys():
            for m in dict_m_pour_e[esp]:
                if len(m)==5: #TODO : fix F090
                    set_milieu.add(m) 
    return set_milieu

@st.cache_data
def get_df_Jaccard(set_r, set_m, e_pour_m):
    list_res = list()
    
    for mil in set_m:
        i_j = indice_jaccard(set_r, set(e_pour_m[mil]['espèces']))
        list_res.append([dict_milieux[mil], i_j[1], i_j[0]])
    
    df_J = pd.DataFrame(list_res, 
                        index=list(set_m), 
                        columns=["Nom", "Indice de Jaccard", "nbr espèces observées"])
    
    return df_J.sort_values(by="Indice de Jaccard", axis=0, ascending=False)

@st.cache_data
def get_df_fact(df_f, df_r):
    df_merge = pd.merge(df_f, df_r,how='inner', on='Nom flore')
    return df_merge[["Nom flore", "Lumiere", "Temperature", "Humidite_edaphique", "Reaction_du_sol_(pH)", "Niveau_trophique", "Matiere_organique", "Salinite", "Texture", "Continentalite", "Code_Releve"]]
    

@st.cache_data
def atypicite_moyenne_releves(data):
    return data[["Code_Releve", "Atypicité"]].groupby(["Code_Releve"]).mean()["Atypicité"]



if __name__ == "__main__":

    tab1, tab2 = st.tabs(["Visu&Annotation", "Statistiques"])

    ###################################################
    # Chargement des donnees
    # msg = st.toast("Chargement des données...")
    data, observateurs, especes = load_data(DATA_PATH)
    dict_milieux, milieu_pour_chaque_espece, especes_pour_chaque_milieu = load_Villaret(VILLARET_PATH)
    data_fact_abiotiques = load_data_fact_abiotiques(FACT_PATH)

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
        col_carte, col_annot, col_meta = st.columns([5, 3, 2], border= True, gap=None) # separation de l'affichage en 3 : une partie pour la carte, une pour l'annotation et une pour les metadonnees

        ###################################################
        # Selection des filtres
        
        st.sidebar.subheader("Filtres") # menu de selection des filtres
        with st.sidebar.form(key="filtres"):
            st.session_state.filters = dict()
            st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs, placeholder="Aucune sélection")
            st.session_state.filters[species_column] = st.multiselect("Espèce", especes, placeholder="Aucune sélection")
            st.session_state.filters["ID"] = st.number_input("ID", min_value=0, value=None, step=1)
            st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["lo_Score"], st.session_state.filters["hi_Score"] = st.select_slider("Atypicité", options=[i for i in np.arange(0, 10.5, 0.5)], value=(0,10))
            st.session_state.filters['Top_atypicity'] = st.slider('Filter les plus atypiques', min_value=5, max_value=100, value=20, step=5)
            # st.markdown('''0 :green[----------]:yellow[----------]:orange[----------]:red[----------]:violet[----------] 10''') # légende
            st.session_state.filters["Méthode"] = st.radio("Méthode de calcul de l'atypicité :", ["rank_ground_truth"])

            st.session_state.filtered = st.form_submit_button(label="Enregistrer") # validation des filtres
            if st.session_state.filtered : #creation d'un subset des donnees filtrees
                with st.sidebar.status("Selection des données...") as status:
                    st.session_state.filtered_data = filter_data(data, st.session_state.filters)
                    status.update(label='Données filtrées', state = "complete")

                with st.sidebar.status("Calcul de l'atypicité sur les données filtrées...") as status:
                    st.session_state.filtered_data["Atypicité"] = compute_atypicity(st.session_state.filtered_data, 
                                                                                    data, st.session_state.filters["Méthode"])
                    status.update(label='Atypicité calculée', state = "complete")
                

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
                st.error("Aucune observation ne correspond à ces critères")
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
        if type(st.session_state.filtered_data) == type(None):
            st.write("Veuillez filtrer les données")
        elif type(st_data1['last_object_clicked']) == type(None): # si aucune observation n'a ete selectionnee
            st.write("Veuillez cliquer sur une observation pour afficher les données associées")
        else : 

            col_1, col_2, col_3 = st.columns([1, 1, 1], border= True, gap=None)
            
            
            ###################################################
            # Colonne pour les données spatiales
            with col_1 :
                st.subheader("Données spatiales")
                afficher_stats_geo(st.session_state.filtered_data, 
                            st.session_state.id_obs, 
                            st.session_state.output_data)
                
                # st.write(f"**Carte des observations** de {data.at[st.session_state.id_obs, species_column]} :")
                st.text(f"Carte des observations de {data.at[st.session_state.id_obs, species_column]} :",
                          help = "L'ensemble des observations disponibles pour l'espèce sont affichées. La couleur des points correspond à leur atypicité. Il n'est pas possible de sélectionner une observation depuis cet affichage.")
                
                map2, group2 = make_map(data.loc[data[species_column]==data.at[st.session_state.id_obs, species_column]],
                                colormap,
                                annotated= st.session_state.output_data['ID'].to_list(),
                                toggle_clusters=False, 
                                toggle_dpt=st.session_state.dpt)

                st_data2 = st_folium(map2, key='map2', 
                                     width=__width__, height=__height__)
                    
            ###################################################
            # Colonne pour les données de fréquence et d'atypicité       
            with col_2 :
                st.subheader("Données de fréquence")
                afficher_stats_atypicite(data, 
                            st.session_state.id_obs, 
                            st.session_state.output_data)
                
                st.write("")
                
                ###################################################
                # Histogramme des atypicités de l'espèce
                st.text(f"Histogramme des atypicités des spécimens de {data.at[st.session_state.id_obs, species_column]} :",
                        help="La ligne rouge correspond à l'observation en cours d'affichage. Les colonnes bleues correspondent au nombre d'observations pour une valeur d'atypicité donnée")

                fig, ax = plt.subplots()
                ax.hist(data.loc[data[species_column]==data.at[st.session_state.id_obs, species_column]]["Atypicité"],
                        bins=10, color="blue")
                plt.axvline(x=data.at[st.session_state.id_obs, 'Atypicité'], color="red", label=f"Atypicité(observation) = {round(data.at[st.session_state.id_obs, 'Atypicité'], 3)}")
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
                at_1 = data.loc[data["Code_Releve"]==data.at[st.session_state.id_obs, "Code_Releve"]]["Atypicité"]
                at_tous = atypicite_moyenne_releves(data)
                
                fig, ax = plt.subplots()
                ax.boxplot([at_1, at_tous])
                ax.set_xticklabels([f"Profil du relevé n°{data.at[st.session_state.id_obs, 'Code_Releve']} \n (Intra)", 
                                    "Profil de l'ensemble des relevés \n (Inter)"])
                plt.ylabel("Atypicité")
                st.pyplot(fig)
                
                
            ###################################################
            # Colonne pour les données relatives au relevé et au milieu 
            with col_3 :
                st.subheader("Données du relevé")
                
                afficher_metadonnees_releve(data, st.session_state.id_obs)
                
                # dataframe et sets contenant les infos du relevé
                df_releve, set_releve = get_releve(data, data.at[st.session_state.id_obs, 'Code_Releve'])
                set_milieu = get_set_milieu(set_releve, milieu_pour_chaque_espece)
                
                ###################################################
                # Tableau des observations
                st.write("**Observations du relevé** :")
                st.dataframe(df_releve, 
                             hide_index=True, 
                             column_order=("ID", species_column, "Nom_Valide", "NbObs", "Atypicité"),
                             height = min(200, len(df_releve)*60))
                

                st.write("")
                
                ###################################################
                # Tableau des indices de Jaccard
                df_Jaccard = get_df_Jaccard(set_releve, set_milieu, especes_pour_chaque_milieu)
                
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

                df_fact_abiotiques = get_df_fact(data_fact_abiotiques, df_releve)
                
                if len(df_fact_abiotiques)!=0:
                    st.text("Profil écologique :",
                            help = "Les indices d'Ellenberg mesurent les préférences d'une espèce pour différents facteurs abiotiques. Synthétiser les indices d'Ellenberg du relevé peut permettre de comprendre le type de milieu qu'il représente.")
                    
                    list_box = list()
                    list_col_Ellenberg = ['Lumiere', 'Temperature', "Humidite_edaphique", "Reaction_du_sol_(pH)", "Matiere_organique", "Texture", 'Salinite']
                    for c in list_col_Ellenberg:
                        list_box.append(df_fact_abiotiques[c])
                    
                    fig, ax = plt.subplots()
                    ax.boxplot(list_box)
                    ax.grid(axis="both", color="lightgrey", linewidth=0.7)
                    plt.xticks([i for i in range(1,len(list_col_Ellenberg)+1)], 
                               labels=[i.replace("_", "\n", 1) for i in list_col_Ellenberg], 
                               rotation=60)
                    st.pyplot(fig)
                else :
                    st.text("Profil écologique :",
                            help = "Les données utilisées ne contiennent pas toutes les espèces présentes en Isère. Aucune information n'a été trouvée concernant les préférences écologiques des espèces du relevé")
                    st.markdown("-> observations insuffisantes pour calculer le profil écologique", unsafe_allow_html=True)
                
                
            ###################################################
            # Affichage d'un tableau de donnees supplementaire
    
            st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
            st.dataframe(st.session_state.filtered_data.head(100), 
                        hide_index=True,
                        column_order=("ID", species_column, "Nom_Valide", "Latitude", "Longitude", "PrenomNom", "NbObs", "Groupe", "Atypicité", "rank_ground_truth", "Code_Releve", "Date_Releve", "NbObs_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation"))


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
    #             # Annotation
                
    #             with row2 :
    #                 st.subheader("Annotation")
    #                 actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Modifier la position", "Signaler un micro-milieux", "Valider l'observation", "Autre"]
    #                 action = st.selectbox("Que souhaitez-vous faire ?", actions_possibles, index=None, placeholder="Veuillez choisir une option")
    #                 annoter(data, action, st.session_state.id_obs, especes)
                    
    #             # TODO : ajouter code postal/commune, nbr de communes où l'espèce est présente
    #             # TODO : superposition cartes
                
