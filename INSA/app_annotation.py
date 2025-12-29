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


st.set_page_config(layout="wide")

# TODO: renvoyer les chemins d'accès, paramètres, constants, etc. dans un fichier de config séparé (en .yml)
path2param = Path(__file__).parent / "params.yml"

print(f"Loading parameter file: {path2param.resolve()}")
yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
params = yaml.load(path2param)
DATA_PATH = params['DATA_PATH']
ISERE = params['ISERE']
__width__ = params['width']
# DATA_PATH = "../../result_export.csv"
print(f"Data path: {Path(DATA_PATH).resolve()}\n")

__GRENOBLE__ = (45.0106, 9.4330)
colormap = cm.LinearColormap(["green", "yellow", "red", "purple"], vmin=0, vmax=10, caption="échelle d'atypicité")

@st.cache_data
def compute_atypicity(filtered_data, data, method):
    """
    Ajoute une colonne "score d'atypicité" aux données

    Parameters
    ----------
    filtered_data : pandas data frame
        tableau contenant les données filtrées

    data : pandas data frame
        tableau contenant les données non filtrées 
        (note: nécessaire pour avoir une échelle cohérente quel que soit le sous-échantillon. Evite aussi une division par zéro si une seule valeur de rang)
        
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
            return 10*(filtered_data["rank_ground_truth"]-np.min(data["rank_ground_truth"]))/(np.max(data["rank_ground_truth"])-np.min(data["rank_ground_truth"]))

# TODO: create an sqlite database with password by user, and propose the option to load only user specific data
# Using sqlite would allow to make the loading faster, with a table with only in lat/lon/date/species/observer data, and other tables with other metadata
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

    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore", "NbObs", "Nom_Valide", "Groupe", "Date_Releve", "Code_Releve", "NbObs_Releve"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data["ID"] = data.index # on rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique
    data["Atypicité"] = compute_atypicity(data, data, "rank_ground_truth")
    observateurs = list(data["PrenomNom"].unique())
    especes = list(data["Nom flore"].unique())
    for i in ["espece", "longitude", "latitude", "micro", "remarque"]:
        data[f"annotation_{i}"] = None
    if not "validation" in data.columns: # do not erase existing validation annotations
        data["validation"] = False
    else:
        data["validation"] = data["validation"].fillna(False)
        if data["validation"].dtype != bool:
            st.error("La colonne 'validation' doit contenir des valeurs booléennes (True/False). Veuillez corriger le fichier de données.")
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
        
        # filtered_data["Atypicité"] = compute_atypicity(filtered_data, data, st.session_state.filters["Méthode"])
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]<st.session_state.filters["hi_Score"]]
        filtered_data = filtered_data.loc[filtered_data["Atypicité"]>st.session_state.filters["lo_Score"]]
        filtered_data = filtered_data.sort_values(by="Atypicité", ascending=False).head(int(st.session_state.filters['Top_atypicity']))
    return filtered_data

# TODO: change appearance of markers when data is validated.
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
        
        for i in range(N): # on affiche les N marqueurs
            point = df.iloc[i]
            folium.CircleMarker(
                location=list(df.iloc[i].loc[['Latitude', 'Longitude']]),
                radius=7,
                color="black",
                fill=True,
                fill_color=colormap(float(point.loc[['Atypicité']].iloc[0])),
                fill_opacity=1,
                popup=f"ID : {point.loc[['ID']].iloc[0]}<br> espèce : {point.loc[['Nom flore']].iloc[0]}<br> atypicité : {round(point.loc[['Atypicité']].iloc[0], 3)}", 
                hover=True, 
                tooltip=f"ID : {point.loc[['ID']].iloc[0]}<br> espèce : {point.loc[['Nom flore']].iloc[0]}<br> atypicité : {round(point.loc[['Atypicité']].iloc[0], 3)}"
                ).add_to(group)

# TODO: compute also geographical span of data to propose a zoom level
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

def compute_bounds(data):
    """
    Calcule les limites géographiques des données

    Parameters
    ----------
    data : pandas data frame
        tableau contenant les données filtrées
        
    Returns
    -------
    bounds : list
        liste des coordonnées des coins sud-ouest et nord-est des données
    """  
    sw = (data["Latitude"].min(), data["Longitude"].min())
    ne = (data["Latitude"].max(), data["Longitude"].max())
    return [sw, ne]

# todo: add legend
def make_map(f_data, colormap, N=30, toggle_clusters=False, toggle_dpt=False, 
             center=__GRENOBLE__, zoom_start=8):
    """
    Ajoute les N premiers marqueurs sur la carte, parmi les données filtrées

    Parameters
    ----------
    f_data : pandas data frame
        tableau contenant les données filtrées
        
    colormap : branca colormap
        échelle de couleur du vert au violet pour représenter l'atypicité
        
    N : int
        nombre d'observations à afficher sur la carte

    toggle_clusters : boolean
        option pour afficher les observations sous forme condensée ou pas      
        
    Returns
    -------
    st_folium
        carte
    """
    try:
        center, zoom_start = compute_center(f_data)
    except:
        pass
    map_ = folium.Map(location=center, zoom_start=zoom_start) #affiche la carte centrée sur Grenoble
    if st.session_state.dpt: folium.GeoJson(ISERE).add_to(map_)   
    if toggle_clusters : # affichage groupé des observations
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1 = folium.FeatureGroup("observations").add_to(marker_cluster)
    else :
        group_1 = folium.FeatureGroup("observations").add_to(map_)
    add_markers(f_data, colormap, group_1, N=N)

    return map_
    # return st_folium(map_, width=width, height=height, key=key, on_change=callback)

def update_id_obs(st_data, current, last):
    new = int(st_data['last_object_clicked_popup'].split()[2])
    if new == current:
        return (current, last)
    else :
        return (new, current)

# TODO: change it to display it in a more compact way (for instance on mouse click)
def afficher_metadonnees(data, id_obs):
    # index_in_filtered_data = int(data["ID"].to_list().index(id_obs))
    st.write(f"ID : {id_obs}")
    st.write(f"espèce : {data.at[id_obs, 'Nom flore']}")
    st.write(f"nom valide : {data.at[id_obs, 'Nom_Valide']}")
    if data.at[id_obs, 'annotation_espece']:
        st.write(f":green[espèce corrigée : {data.at[id_obs, 'annotation_espece']}]")
    st.write(f"groupe : {data.at[id_obs, 'Groupe']}")
    st.write(f"observateur : {data.at[id_obs, 'PrenomNom']}")
    st.write(f"date : {pd.to_datetime(data.at[id_obs, 'Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")
    st.write(f"coordonnées : ({data.at[id_obs, 'Latitude']}, {data.at[id_obs, 'Longitude']})")
    if data.at[id_obs, 'annotation_latitude'] or data.at[id_obs, 'annotation_longitude']: 
        st.write(f":green[coordonnées corrigées : {data.at[id_obs, 'annotation_remarque']}]")
    st.write(f"spécimens observés : {int(data.at[id_obs, 'NbObs'])}")
    st.write(f"atypicité : {round(data.at[id_obs, 'Atypicité'], 3)}")
    if data.at[id_obs, 'annotation_remarque']:
        st.write(f":green[remarque : {data.at[id_obs, 'annotation_remarque']}]")
    if data.at[id_obs, 'annotation_remarque'] or data.at[id_obs, 'annotation_espece'] or data.at[id_obs, 'annotation_micro'] or data.at[id_obs, 'annotation_latitude'] or data.at[id_obs, 'annotation_longitude']:
        st.write(":green[observation annotée]")
    elif data.at[id_obs, 'validation']:
        st.write(":green[observation validée]")
    else: 
        st.write(":red[observation en attente de validation]")

def afficher_metadonnees_releve(data, id_obs):
    st.write("------")
    st.write(f"code relevé : {data.at[id_obs, 'Code_Releve']}")
    st.write(f"date relevé : {data.at[id_obs, 'Date_Releve']}")
    st.write(f"observations dans le relevé : {data.at[id_obs, 'NbObs_Releve']}")

def annoter(data, action, id_obs, especes):
    if action!=None:
        with st.form(clear_on_submit=True, key="annotation"):       
            match action:
                case "Modifier l'espèce/le nom de l'espèce":
                    data.at[id_obs, "annotation_espece"] = st.selectbox("Nom de l'espèce", especes)
                case "Modifier la position":
                    data.at[id_obs, "annotation_latitude"] = st.text_input("Latitude", "")
                    data.at[id_obs, "annotation_longitude"] = st.text_input("Longitude", "")
                case "Signaler un micro-milieux":
                     data.at[id_obs, "annotation_micro"] = st.text_area("Description", "")
                case "Valider l'observation":
                    data.at[id_obs, "annotation_validation"] = st.checkbox("Je confirme l'observation")
                case "Autre":
                    data.at[id_obs, "annotation_remarque"] = st.text_area("Autres remarques", "")
            st.form_submit_button(label="Enregistrer") # validation de l'annotation
            

if __name__ == "__main__":
    st.title("Outil d'annotation")

    # TODO: use only a single tab both for visulisation and annotation, with one more pane.
    # Metadata are not that important, they can be displayed on click, and save some space in display.
    
    tab1, tab2 = st.tabs(["Visualisation", "Annotation"])

    ###################################################
    # Chargement des donnees
    # msg = st.toast("Chargement des données...")
    data, observateurs, especes = load_data(DATA_PATH)

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
        col_carte, col_data = st.columns([3, 1], border= True) # separation de l'affichage en 2 : une partie pour la carte et une pour les metadonnees

        ###################################################
        # Selection des filtres
        
        st.sidebar.subheader("Filtres") # menu de selection des filtres
        with st.sidebar.form(key="filtres"):
            st.session_state.filters = dict()
            st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs, placeholder="Aucune sélection")
            st.session_state.filters["Nom flore"] = st.multiselect("Espèce", especes, placeholder="Aucune sélection")
            st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
            st.session_state.filters["lo_Score"], st.session_state.filters["hi_Score"] = st.select_slider("Atypicité", options=[i for i in np.arange(0, 10.5, 0.5)], value=(0,10))
            st.session_state.filters['Top_atypicity'] = st.slider('Filter les plus atypiques', min_value=5, max_value=100, value=20, step=5)
            st.markdown('''0 :green[----------]:yellow[----------]:orange[----------]:red[----------]:violet[----------] 10''') # légende
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
        st.session_state.dpt =st.sidebar.toggle("Afficher le département de l'Isère")



        # TODO: add statistics on number of obs, number of filtered obs, number of species, histogram on Atypicity


        ###################################################
        # Affichage de la carte

        with col_carte:
            sub_col_carte_1, sub_col_carte_2 = st.columns(2)
            
            sub_col_carte_1.subheader("Carte des observations")
            if type(st.session_state.filtered_data) != type(None): # si les donnees ont ete filtrees par l'utilisateur
                st.session_state.filters["N"] = sub_col_carte_2.slider("Combien d'observations afficher ?", min_value=1, max_value=100, value=30, step=1)
            else :
                st.session_state.filters["N"] = 50
            
            if type(st.session_state.filtered_data)!=type(None) and len(st.session_state.filtered_data) == 0:
                st.error("Aucune observation ne correspond à ces critères")
            else :
            # FIXME: make the map load faster
                map1 = make_map(st.session_state.filtered_data,
                                colormap,
                                N=st.session_state.filters["N"],
                                toggle_clusters=clusters, 
                                toggle_dpt=st.session_state.dpt)

                st_data1 = st_folium(map1, key='map1', 
                                     width=__width__)
        
        ###################################################
        # Affichage des metadonnees

        with col_data:
            st.subheader("Metadonnées")
            if type(st.session_state.filtered_data) == type(None):
                st.write("Veuillez filtrer les données")
            elif type(st_data1['last_object_clicked_popup']) == type(None): # si aucune observation n'a ete selectionnee
                st.write("Veuillez cliquer sur une observation pour afficher les données associées")
            else : 
                st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, st.session_state.id_obs, st.session_state.last)
                afficher_metadonnees(st.session_state.filtered_data, st.session_state.id_obs)

        ###################################################
        # Affichage d'un tableau de donnees supplementaire
        # TODO : lier le tableau à la carte et aux metadonnees pour sélectionner un point
        
        if type(st.session_state.filtered_data) != type(None):
            st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
            select_row = st.dataframe(st.session_state.filtered_data.head(st.session_state.filters["N"]), 
                        hide_index=True, 
                        selection_mode="single-row",
                        on_select="rerun",
                        column_order=("ID", "Nom flore", "Nom_Valide", "Latitude", "Longitude", "PrenomNom", "NbObs", "Groupe", "Atypicité", "rank_ground_truth", "Code_Releve", "Date_Releve", "NbObs_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation"))
                
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
    #                     st_data2 = make_map(data.loc[data["Nom flore"]==data.at[st.session_state.id_obs, 'Nom flore']],
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
    #                     ], hide_index=True, column_order=("ID", "Nom flore", "Nom_Valide", "NbObs", "Atypicité"))
                    
    #             # TODO : ajouter code postal/commune, nbr de communes où l'espèce est présente
    #             # TODO : superposition cartes
                
