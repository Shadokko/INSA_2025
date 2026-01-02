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
from INSA.templates import add_template2map
import time

st.set_page_config(layout="wide")

# TODO: renvoyer les chemins d'accès, paramètres, constants, etc. dans un fichier de config séparé (en .yml)
path2param = Path(__file__).parent / "params.yml"

print(f"Loading parameter file: {path2param.resolve()}")
yaml=YAML(typ='safe')   # default, if not specfied, is 'rt' (round-trip)
params = yaml.load(path2param)
DATA_PATH = params['DATA_PATH']
print(f"Data path: {Path(DATA_PATH).resolve()}\n")

export_path = Path(params['export_path'])
print(f"Export path: {export_path.resolve()}\n")

ISERE = params['ISERE']
__width__ = params['width']
species_column = params['species_column']


# DATA_PATH = "../../result_export.csv"


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

    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", species_column, "NbObs", "Nom_Valide", "Groupe", "Date_Releve", "Code_Releve", "NbObs_Releve"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0) # on fusionne tous les paquets pour obtenir les données complètes
    data["ID"] = data.index # on rajoute une colonne ID qui nous permettra d'identifier chaque ligne de façon unique
    data["Atypicité"] = compute_atypicity(data, data, "rank_ground_truth")
    observateurs = list(data["PrenomNom"].unique())
    especes = list(data[species_column].unique())
    for i in ["espece", "longitude", "latitude", "micro", "remarque"]:
        if not f"annotation_{i}" in data.columns: # do not erase existing validation annotations
            data[f"annotation_{i}"] = None
    if not "validation" in data.columns: # do not erase existing validation annotations
        data["validation"] = None
    else:
        data["validation"] = data["validation"].fillna(None)
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

        if st.session_state.id_obs == int(row['ID']): color="red"
        else: color='black'

        folium.CircleMarker(location=list(row.loc[['Latitude', 'Longitude']]),
                            radius=radius,
                            color=color,
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
    if type(st_data['last_object_clicked_popup']) == type(None):
        return (current, last)
    else:
        new = int(st_data['last_object_clicked_popup'].split()[2])
        if new == current:
            return (current, last)
        else :
            return (new, current)

@st.cache_data
def afficher_metadonnees(data, id_obs, output_data):
    """
    Affiche les métadonnées associées à une observation
    """
    
    # index_in_filtered_data = int(data["ID"].to_list().index(id_obs))
    lines = []
    lines.append(f"ID : {id_obs}")

    if id_obs in output_data['ID'].to_list():
        lines.append(":green[observation annotée]")
        lines.append(f"Statut: :green[{output_data.loc[output_data['ID']==id_obs, 'validation'].values[0]}]")

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
    lines.append(f"date : {pd.to_datetime(data.at[id_obs, 'Date_Releve'],format='%Y-%m-%d').strftime('%d %B %Y')}")
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
def afficher_metadonnees_releve(data, id_obs):
    st.write("------")
    st.write(f"code relevé : {data.at[id_obs, 'Code_Releve']}")
    st.write(f"date relevé : {data.at[id_obs, 'Date_Releve']}")
    st.write(f"observations dans le relevé : {data.at[id_obs, 'NbObs_Releve']}")


actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Signaler un micro-milieux"]
def annoter(data, id_obs, especes):
    if action!=None:
        match action:
            case "Modifier l'espèce/le nom de l'espèce":
                st.session_state.output_data.at[id_obs, "annotation_espece"] = st.selectbox("Nom de l'espèce", especes)
    #                 data.at[id_obs, "annotation_espece"] = st.selectbox("Nom de l'espèce", especes)
    #             case "Modifier la position":
    #                 data.at[id_obs, "annotation_latitude"] = st.text_input("Latitude", "")
    #                 data.at[id_obs, "annotation_longitude"] = st.text_input("Longitude", "")
            case "Signaler un micro-milieux":
                st.session_state.output_data.at[id_obs, "annotation_micro"] = st.text_area("Description", "")
    #             case "Valider l'observation":
    #                 data.at[id_obs, "annotation_validation"] = st.checkbox("Je confirme l'observation")
    #             case "Autre":
    #                 data.at[id_obs, "annotation_remarque"] = st.text_area("Autres remarques", "")
    #         st.form_submit_button(label="Enregistrer") # validation de l'annotation

def save_annotations(data, export_path):
    """
    Saves (exports) ALL annotations to a CSV file
    
    :param data: Description
    :param export_path: Description
    """
    export_path = export_path.parent / (export_path.stem + time.asctime().replace(" ", "_").replace(":", "-") + export_path.suffix)
    data.to_csv(export_path, sep=";", index=False)
    st.success(f"Données exportées vers {export_path.resolve()}")

def _save_annotation(id_obs, validation_status, selected_species):
    """
    Save the selected observation to the in-memory output table. (which will later be saved as csv through save_annotations)

    Avoids duplicates
    """
    if id_obs is None:
        st.error("Aucune observation sélectionnée — impossible de sauvegarder.")
        return
    
    st.session_state.output_data = pd.concat([
        st.session_state.output_data,
        data.loc[data["ID"]==id_obs].assign(validation=validation_status, annotation_espece=selected_species)
    ], ignore_index=True).drop_duplicates(subset=['ID'], keep='last')


if __name__ == "__main__":
    # st.sidebar.title("Outil d'annotation")

    tab1, tab2 = st.tabs(["Visu&Annotation", "Statistiques"])

    ###################################################
    # Chargement des donnees
    # msg = st.toast("Chargement des données...")
    data, observateurs, especes = load_data(DATA_PATH)

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
            st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs, placeholder="Aucune sélection")
            st.session_state.filters[species_column] = st.multiselect("Espèce", especes, placeholder="Aucune sélection")
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


        ###################################################

        with col_annot:

            #########################
            # Formulaire d'annotation
                
            if type(st.session_state.filtered_data) == type(None):
                st.write("Veuillez filtrer les données")
            elif st.session_state.id_obs is None: # si aucune observation n'a ete selectionnee
                st.write("Veuillez cliquer sur une observation pour l'annoter")
            else: 
                st.button("Exporter les annotations", on_click=lambda: save_annotations(st.session_state.output_data, export_path))
     
                with st.form(key="annotation"):
                    st.subheader("Annotation de l'observation")
                    # Update the selected id from the map component early so it is preserved across reruns

                    default_espece = st.session_state.filtered_data.at[st.session_state.id_obs, species_column]
                    if default_espece in especes:
                        default_index = especes.index(default_espece)
                    else:
                        default_index = 0
                    st.session_state.selected_species = st.selectbox(f"Modifier l'espèce (initialement: {default_espece})", 
                                                                     especes, index=default_index)

                    st.session_state.validation_status = st.radio("Validation de la donnée:", ['Je confirme', 'Donnée douteuse', "Donnée fausse"])
                    # action = st.selectbox("Que souhaitez-vous faire ?", actions_possibles, index=None, placeholder="Veuillez choisir une option")
                    # annoter(data, action, st.session_state.id_obs, especes)

                    st.form_submit_button("Sauvegarder l'annotation", on_click=_save_annotation, args=(st.session_state.id_obs, 
                                                                                                       st.session_state.validation_status, 
                                                                                                       st.session_state.selected_species))

                # actions_possibles = ["Modifier l'espèce/le nom de l'espèce", "Modifier la position", "Signaler un micro-milieux", "Valider l'observation", "Autre"]
                # action = st.selectbox("Que souhaitez-vous faire ?", actions_possibles, index=None, placeholder="Veuillez choisir une option")
                # annoter(data, action, st.session_state.id_obs, especes)

                ###################################################
                # Affichage des metadonnees
            with col_meta:
                st.subheader("Metadonnées de l'observation")

                # st.session_state.id_obs, st.session_state.last = update_id_obs(st_data1, st.session_state.id_obs, st.session_state.last)
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
                        column_order=("ID", species_column, "Nom_Valide", "Latitude", "Longitude", "PrenomNom", "NbObs", "Groupe", "Atypicité", "rank_ground_truth", "Code_Releve", "Date_Releve", "NbObs_Releve", "annotation_espece", "annotation_latitude", "annotation_longitude", "annotation_micro", "annotation_remarque", "validation"))
                
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
                
