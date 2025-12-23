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


st.set_page_config(layout="wide")

# TODO : fix relative path
# TODO: renvoyer les chemins d'accès, paramètres, constants, etc. dans un fichier de config séparé (en .yml)
path2param = Path(__file__).parent / "params.yml"

print(f"Loading parameter file: {path2param.resolve()}")
yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
params = yaml.load(path2param)
DATA_PATH = params['DATA_PATH']
# DATA_PATH = "../../result_export.csv"
print(f"Data path: {Path(DATA_PATH).resolve()}\n")


GRENOBLE = (45.0106, 9.4330)
colormap = cm.LinearColormap(["green", "yellow", "red", "purple"], vmin=0, vmax=10, caption="échelle d'atypicité")

@st.cache_data
def compute_atypicity(data, method):
    """
    Ajoute une colonne "score d'atypicité" aux données

    Parameters
    ----------
    data : pandas data frame
        tableau contenant les données
        
    method : string
        méthode utilisée pour calculer le score 
        ["rank_ground_truth"]
        
    Returns
    -------
    filtered_data : pandas data frame
        tableau contenant les données filtrées
    """  
    
    match method:
        case "rank_ground_truth":
            return 10*(data["rank_ground_truth"]-np.min(data["rank_ground_truth"]))/(np.max(data["rank_ground_truth"])-np.min(data["rank_ground_truth"]))


@st.cache_data
def load_data(filename):
    """
    Charge les données 

    Parameters
    ----------
    filename : string
        chemon vers les données
        
    Returns
    -------
    data : pandas data frame
        tableau contenant uniquement les données utiles
        
    observateurs : list
        liste des observateurs présents dans les données, sans répétition
    
    especes : list
        liste des espcèces présentes dans les données, sans répétition
    """
    # TODO: would be nice to have either a drag'n'drop option to upload a new data file, or a file selector with a browser
    chunk_size = 10_000 # les données seront chargées par paquets pour aller plus vite
    chunks = [] # liste qui contiendra tous les paquets de données

    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore", "Date_Releve", "Nbre_donnees_espece"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data["ID"] = data.index # on rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique
    data["Atypicité"] = compute_atypicity(data, "rank_ground_truth")
    data["Nbre_donnees_espece"] = data["Nbre_donnees_espece"].str.replace('[', '').str.replace(']', '')
    observateurs = list(data["PrenomNom"].unique())
    especes = list(data["Nom flore"].unique())
    
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
    
    if (len(st.session_state.filters["PrenomNom"]) == 0) and (len(st.session_state.filters["Nom flore"]) == 0): # on vérifie que l'utilisateur a choisi au moins un observateur ou une espèce, sinon il y a trop de données à afficher, et c'est plus logique
        st.error("Veuillez choisir un observateur ou une espèce.") # erreur qui s'affiche si ce n'est pas le cas
    else :
        if len(st.session_state.filters["PrenomNom"]) != 0: # si il y a un filtre sur l'observateur
            filtered_data = filtered_data.loc[[element in filters["PrenomNom"] for element in filtered_data["PrenomNom"]]] # on filtre par rapport à l'observateur
        if len(st.session_state.filters["Nom flore"]) != 0: # si il y a un filtre sur l'espèce
            filtered_data = filtered_data.loc[[element in filters["Nom flore"] for element in filtered_data["Nom flore"]]] # on filtre par rapport à l'espèce
        if st.session_state.filters["Debut"] > st.session_state.filters["Fin"] : # si le dates choisies ne sont pas dans le bon ordre
            st.error("Veuillez choisir une date de début antérieure à la date de fin.") # erreur affichée
        else :
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date >= filters["Debut"]] # sinon, on garde uniquement les données ultérieures à la date de début choisie
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date <= filters["Fin"]] # puis, on garde uniquement les données précédant la date de début choisie
        
        filtered_data["Atypicité"] = compute_atypicity(filtered_data, st.session_state.filters["Méthode"])
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]<st.session_state.filters["hi_Score"]]
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]>st.session_state.filters["lo_Score"]]
    return filtered_data


def add_markers(df, colormap, group, N=0):
    """
    Ajoute les N premiers marqueurs sur la carte, parmi les données filtrées

    Parameters
    ----------
    df : pandas data frame
        tableau contenant les données filtrées
        
    colormap : branca colormap
        échelle de couleur du vert au violet pour représenter l'atypicité
        
    group : folium.FeatureGroup()
        groupe de marqueurs
        
    N : int
        nombre d'observations à afficher sur la carte
        
    Returns
    -------
    filtered_data : pandas data frame
        tableau contenant les données filtrées
    """  
    
    if type(df) != type(None): #si le dataframe n'est pas vide
        if N == 0: # si le nombre d'observations à afficher est celui par défaut
            N = len(df) # alors on affiche toutes les données
        else:
            N = min(N, len(df)) # sinon, on affiche les N premières observations filtrées, ou toutes s'il y en a moins de N
        
        if N == 1:
            folium.CircleMarker(
                location=list(df.loc[['Latitude', 'Longitude']]),
                radius=7,
                color="black",
                fill=True,
                fill_color=colormap(float(df.loc[['Atypicité']].iloc[0])),
                fill_opacity=1,
                popup=df.loc[['ID']].iloc[0]
                ).add_to(group)
        else : 
            for i in range(N): # on affiche les N marqueurs
                folium.CircleMarker(
                    location=list(df.iloc[i].loc[['Latitude', 'Longitude']]),
                    radius=7,
                    color="black",
                    fill=True,
                    fill_color=colormap(float(df.iloc[i].loc[['Atypicité']].iloc[0])),
                    fill_opacity=1,
                    popup=df.index[i]
                    ).add_to(group)


def make_map(f_data, colormap, center, width, height, N, toggle_clusters, key):
    """
    Ajoute les N premiers marqueurs sur la carte, parmi les données filtrées

    Parameters
    ----------
    f_data : pandas data frame
        tableau contenant les données filtrées
        
    colormap : branca colormap
        échelle de couleur du vert au violet pour représenter l'atypicité
        
    center : tuple de float
        coordonnées du centre de la carte
        
    N : int
        nombre d'observations à afficher sur la carte

    toggle_clusters : boolean
        option pour afficher les observations sous forme condensée ou pas      
        
    Returns
    -------
    st_folium
        carte
    """  
    map_ = folium.Map(location=center, zoom_start=8) #affiche la carte centrée sur Grenoble
   
    if toggle_clusters : # affichage groupé des observations
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1 = folium.FeatureGroup("observations").add_to(marker_cluster)
    else :
        group_1 = folium.FeatureGroup("observations").add_to(map_)
    add_markers(f_data, colormap, group_1, N=N)

    return st_folium(map_, width=width, height=height, key=key)

def update_id_obs(st_data, current, last):
    new = int(st_data['last_object_clicked_popup'])
    if new == current:
        return (current, last)
    else :
        return (new, current)

def afficher_metadonnees(data, id_obs, expand=False):
    st.write(f"ID : {id_obs}")
    st.write(f"espèce : {data.at[id_obs, 'Nom flore']}")
    st.write(f"observateur : {data.at[id_obs, 'PrenomNom']}")
    st.write(f"atypicité : {round(data.at[id_obs, 'Atypicité'], 3)}")
    # index_in_filtered_data = int(st.session_state.filtered_data["ID"].to_list().index(st.session_state.id_obs))
    # st.write(f"ID in fd : {index_in_filtered_data}")
    if expand :
        st.write(f"latitude : {data.at[id_obs, 'Latitude']}")
        st.write(f"longitude : {data.at[id_obs, 'Longitude']}")
        st.write(f"date : {pd.to_datetime(data.at[id_obs, 'Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")
        st.write(f"spécimens observés : {data.at[id_obs, 'Nbre_donnees_espece']}")
            
def augmenter_metadonnees(data, id_obs):
    st.write("voici plus d'infos")










st.title("Outil d'annotation")
tab1, tab2 = st.tabs(["Visualisation", "Annotation"])

###################################################
# Chargement des donnees
# msg = st.toast("Chargement des données...")
data, observateurs, especes = load_data(DATA_PATH)
# msg.toast("Données à jour !", icon=":material/check:")

if "filtered" not in st.session_state:
    st.session_state.filtered = False
    st.session_state.filtered_data = None
    st.session_state.id_obs = None
    st.session_state.last = None
    st.session_state.show_more_data = False

###################################################
# Premier onglet pour la visualisation

with tab1:
    col_carte, col_data = st.columns([3, 1], border= True) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees
    
    ###################################################
    # Selection des filtres
    
    st.sidebar.subheader("Filtres") # menu de selection des filtres
    with st.sidebar.form(key="filtres"):
        st.session_state.filters = dict()
        st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs)
        st.session_state.filters["Nom flore"] = st.multiselect("Espèce", especes)
        st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
        st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
        st.session_state.filters["lo_Score"], st.session_state.filters["hi_Score"] = st.select_slider("Atypicité", options=[i for i in np.arange(0, 10.5, 0.5)], value=(0,10))
        st.markdown('''0 :green[----------]:yellow[----------]:orange[----------]:red[----------]:violet[----------] 10''') # légende
        st.session_state.filters["Méthode"] = st.radio("Méthode de calcul de l'atypicité :", ["rank_ground_truth"])
        
        st.session_state.filtered = st.form_submit_button(label="Enregistrer") # validation des filtres
        if st.session_state.filtered : #creation d'un subset des donnees filtrees
            with st.sidebar.status("Selection des données...") as status:
                st.session_state.filtered_data = filter_data(data, st.session_state.filters)
                status.update(label='Données filtrées', state = "complete")
                st.text(st.session_state.filtered_data)
                
    clusters = st.sidebar.toggle("Affichage groupé")
    
    ###################################################
    # Affichage de la carte

    with col_carte:
        sub_col_carte_1, sub_col_carte_2 = st.columns(2)
        
        sub_col_carte_1.subheader("Carte des observations")
        if type(st.session_state.filtered_data) != type(None): # si les donnees ont ete filtrees par l'utilisateur
            st.session_state.filters["N"] = sub_col_carte_2.slider("Combien d'observations afficher ?", min_value=1, max_value=100, value=30, step=1)
        else :
            st.session_state.filters["N"] = 50
            
        # FIXME: make the map load faster
        st_data = make_map(st.session_state.filtered_data, 
                           colormap,
                           GRENOBLE, 
                           2000,
                           500,
                           st.session_state.filters["N"],
                           clusters,
                           key=1)
    
    ###################################################
    # Affichage des metadonnees

    with col_data:
        st.subheader("Metadonnées")
        if type(st.session_state.filtered_data) == type(None):
            st.write("Veuillez filtrer les données")
        elif type(st_data['last_object_clicked_popup']) == type(None): # si aucune observation n'a ete selectionnee
            st.write("Veuillez cliquer sur une observation pour afficher les données associées")
        else : 
            st.session_state.id_obs, st.session_state.last = update_id_obs(st_data, st.session_state.id_obs, st.session_state.last)
            afficher_metadonnees(data, st.session_state.id_obs)

    ###################################################
    # Affichage d'un tableau de donnees supplementaire
    # TODO : lier le tableau à la carte et aux metadonnees pour sélectionner un point
    
    if type(st.session_state.filtered_data) != type(None):
        st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
        select_row = st.dataframe(st.session_state.filtered_data.head(st.session_state.filters["N"]), 
                     hide_index=True, 
                     selection_mode="single-row",
                     on_select="rerun")
            
    else :
        st.subheader("Données brutes")
        st.write("Veuillez filtrer les données")

###################################################
# Deuxième onglet pour l'annotation et l'affichage de données supplémentaires
with tab2:

    if type(st.session_state.filtered_data) == type(None):
        st.write("Veuillez filtrer les données")
    elif type(st.session_state.id_obs) == type(None): # si aucune observation n'a ete selectionnee
        st.write("Veuillez cliquer sur une observation pour afficher les données associées")
    else : 
        col_carte, col_data = st.columns([2, 2], border= True) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees

        with col_carte:
            row1 = st.container(height=400)
            row2 = st.container(height=400)
            
            ###################################################
            # Affichage d'une carte centrée et zoomée sur l'observation
            
            with row1:
                # st.subheader("Carte")
                st_data = make_map(data.iloc[st.session_state.id_obs], 
                               colormap,
                               (st_data['last_object_clicked']["lat"], st_data['last_object_clicked']["lng"]), 
                               360,
                               360,
                               1,
                               clusters,
                               key=2)
                
            ###################################################
            # Annotation
            
            with row2 :
                st.subheader("Annotation")
                
        ###################################################
        # Afficahge de données supplémentaires

        with col_data:
            sub_col_data_1, sub_col_data_2 = st.columns(2)
            sub_col_data_1.subheader("Metadonnées")
            
            if type(st.session_state.filtered_data) != type(None): # si les donnees ont ete filtrees par l'utilisateur
                st.session_state.show_more_data = sub_col_data_2.toggle("Afficher plus de données")

            afficher_metadonnees(data, st.session_state.id_obs, expand=True)
            if st.session_state.show_more_data :
                augmenter_metadonnees(data, st.session_state.id_obs)
            # TODO : rajouter un bouton "augmenter métadonnées" -> code postal/commune, fréquence...
            # TODO : option "montrer les autres observations de cette espèce"
            # TODO : superposition cartes
            # TODO : bouton "modifier cette observation"
            
