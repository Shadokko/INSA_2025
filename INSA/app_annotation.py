import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os

# import datetime
# import branca.colormap as cm


st.set_page_config(layout="wide")

# TODO : fix relative path
# TODO: renvoyer les chemins d'accès, paramètres, constants, etc. dans un fichier de config séparé (en .yml)
DATA_PATH = "../../result_export.csv"
GRENOBLE = (45.110600, 5.433000)
#color_map = cm.LinearColormap(["green", "yellow", "red"],vmin=min(data["rank_ground_truth"]), vmax=max(data["rank_ground_truth"]))
        

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
    chunk_size = 10_000 # les données seront chargées par paquets pour aller plus vite
    chunks = [] # liste qui contiendra tous les paquets de données
    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore", "Date_Releve"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data["ID"] = data.index # on rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique
    
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
    return filtered_data


def add_markers(df, group, N=0):
    """
    Ajoute les N premiers marqueurs sur la carte, parmi les données filtrées

    Parameters
    ----------
    df : pandas data frame
        tableau contenant les données filtrées
        
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
            
        for i in range(N): # on affiche les N marqueurs
            marker = folium.Marker(
                location = df.iloc[i].loc[['Latitude', 'Longitude']],
                popup = df.index[i],
                tooltip = "cliquez pour afficher",
                icon = folium.Icon(color = "blue", 
                                   icon = "leaf", 
                                   prefix = "fa")
                )
            marker.add_to(group)

def make_map(f_data, center, N, toggle_clusters):
    """
    Ajoute les N premiers marqueurs sur la carte, parmi les données filtrées

    Parameters
    ----------
    f_data : pandas data frame
        tableau contenant les données filtrées
        
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
    map_ = folium.Map(location=(45.0106, 9.4330), zoom_start=8) #affiche la carte centrée sur Grenoble
    if toggle_clusters : # affichage groupé des observations
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1 = folium.FeatureGroup("surbrillance").add_to(marker_cluster)
        group_2 = folium.FeatureGroup("autres").add_to(marker_cluster)
    else :
        group_1 = folium.FeatureGroup("surbrillance").add_to(map_)
        group_2 = folium.FeatureGroup("autres").add_to(map_)
    add_markers(f_data, group=group_2, N=N)
    return st_folium(map_, width=2000, height=500)


st.title("Outil d'annotation")
col_carte, col_data = st.columns([3, 1], border= True) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees


###################################################
# Chargement des donnees
with st.sidebar.status("Chargement des données...") as status:
    data, observateurs, especes = load_data(DATA_PATH)
    status.update(label='Données à jour', state = "complete")

if "filtered" not in st.session_state:
    st.session_state.filtered = False
    st.session_state.filtered_data = None


###################################################
# Selection des filtres
st.sidebar.subheader("Filtres") # menu de selection des filtres
with st.sidebar.form(key="filtres2", ):
    st.session_state.filters = dict()
    st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs)
    st.session_state.filters["Nom flore"] = st.multiselect("Espèce", especes)
    st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
    st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
    st.session_state.filters["A_Score"] = st.slider("Atypicité", min_value=0, max_value=10, value=0, step=1)

    st.session_state.filtered = st.form_submit_button(label="Enregistrer") # validation des filtres
    if st.session_state.filtered : #creation d'un subset des donnees filtrees
        with st.sidebar.status("Selection des données...") as status:
            st.session_state.filtered_data = filter_data(data, st.session_state.filters)
            status.update(label='Données filtrées', state = "complete")
            st.text(st.session_state.filtered_data)


###################################################
# Affichage de la carte
clusters = st.sidebar.toggle("Affichage groupé")

with col_carte:
    sub_col_carte_1, sub_col_carte_2 = st.columns(2)
    
    sub_col_carte_1.subheader("Carte des observations")
    if type(st.session_state.filtered_data) != type(None): # si les donnees ont ete filtrees par l'utilisateur
        st.session_state.filters["N"] = sub_col_carte_2.slider("Combien d'observations afficher ?", min_value=0, max_value=len(st.session_state.filtered_data), value=50, step=50)
    else :
        st.session_state.filters["N"] = 50
        
    st_data = make_map(st.session_state.filtered_data, 
                       GRENOBLE, 
                       st.session_state.filters["N"],
                       clusters)

###################################################
# Affichage des metadonnees
with col_data:
    st.subheader("Metadonnées")
    if type(st_data['last_object_clicked_popup']) == type(None): # si aucune observation n'a ete selectionnee
        st.write("Veuillez cliquer sur une observation pour afficher les données associées")
    else : 
        id_obs = int(st_data['last_object_clicked_popup'])
        st.write(f"ID : {id_obs}")
        st.write(f"espèce : {data.at[id_obs, 'Nom flore']}")
        st.write(f"observateur : {data.at[id_obs, 'PrenomNom']}")
        st.write(f"coordonnées : ({data.at[id_obs, 'Latitude']}, {data.at[id_obs, 'Longitude']}")


###################################################
# Affichage d'un tableau de donnees supplementaire
# TODO : lier le tableau à la carte et aux metadonnees pour sélectionner un point
if type(st.session_state.filtered_data) != type(None):
    st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
    select_row = st.dataframe(st.session_state.filtered_data.head(st.session_state.filters["N"]), 
                 hide_index=True, 
                 selection_mode="single-row",
                 on_select="rerun")
    # st.session_state.filtered_data.iloc[select_row.selection["rows"][0]]
    # if type(st_data['last_object_clicked_popup']) != type(None):
        
else :
    st.subheader("Données brutes")





