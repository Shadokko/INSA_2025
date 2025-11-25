from pydoc import doc
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
from pathlib2 import Path
from ruamel.yaml import YAML



# import datetime
# import branca.colormap as cm


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


GRENOBLE = (45.110600, 5.433000)
#color_map = cm.LinearColormap(["green", "yellow", "red"],vmin=min(data["rank_ground_truth"]), vmax=max(data["rank_ground_truth"]))
        

@st.cache_data
def load_data(filename):
    chunk_size = 10_000
    chunks = []
    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore", "Date_Releve"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0)
    data["ID"] = data.index
    
    observateurs = list(data["PrenomNom"].unique())
    especes = list(data["Nom flore"].unique())
    
    return data, observateurs, especes


@st.cache_data
def filter_data(data, filters):
    filtered_data = data.copy()
    
    if (len(st.session_state.filters["PrenomNom"]) == 0) and (len(st.session_state.filters["Nom flore"]) == 0):
        st.error("Veuillez choisir un observateur ou une espèce.")
    else :
        if len(st.session_state.filters["PrenomNom"]) != 0:
            filtered_data = filtered_data.loc[[element in filters["PrenomNom"] for element in filtered_data["PrenomNom"]]]
        if len(st.session_state.filters["Nom flore"]) != 0:
            filtered_data = filtered_data.loc[[element in filters["Nom flore"] for element in filtered_data["Nom flore"]]]
                             
        if st.session_state.filters["Debut"] > st.session_state.filters["Fin"] :
            st.error("Veuillez choisir une date de début antérieure à la date de fin.")           
        else :
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date >= filters["Debut"]]
            filtered_data = filtered_data.loc[pd.to_datetime(filtered_data["Date_Releve"],format='%Y-%m-%d').dt.date <= filters["Fin"]]
    return filtered_data


def add_markers(df, where, N=0):
    if type(df) != type(None):
        if N == 0:
            N = len(df)
        else:
            N = min(N, len(df))
            
        for i in range(N):
            marker = folium.Marker(
                location = df.iloc[i].loc[['Latitude', 'Longitude']],
                popup = df.index[i],
                tooltip = "cliquez pour afficher",
                icon = folium.Icon(color = "blue", 
                                   icon = "leaf", 
                                   prefix = "fa")
                )
            marker.add_to(where)

def make_map(f_data, center, N, toggle_clusters):
    map_ = folium.Map(location=(45.0106, 9.4330), zoom_start=8)    
    if toggle_clusters :
        marker_cluster = folium.plugins.MarkerCluster().add_to(map_)
        group_1 = folium.FeatureGroup("surbrillance").add_to(marker_cluster)
        group_2 = folium.FeatureGroup("autres").add_to(marker_cluster)
    else :
        group_1 = folium.FeatureGroup("surbrillance").add_to(map_)
        group_2 = folium.FeatureGroup("autres").add_to(map_)
    add_markers(f_data, where=group_2, N=N)
    return st_folium(map_, width=2000, height=500)


st.title("Outil d'annotation")

col1, col2 = st.columns([3, 1], border= True)

with st.sidebar.status("Chargement des données...") as status:
    
    data, observateurs, especes = load_data(DATA_PATH)
    status.update(label='Données à jour', state = "complete")

if "filtered" not in st.session_state:
    st.session_state.filtered = False
    st.session_state.filtered_data = None

st.sidebar.subheader("Filtres")
with st.sidebar.form(key="filtres2", ):
    st.session_state.filters = dict()
    st.session_state.filters["PrenomNom"] = st.multiselect("Nom de l'observateur", observateurs)
    st.session_state.filters["Nom flore"] = st.multiselect("Espèce", especes)
    st.session_state.filters["Debut"] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
    st.session_state.filters["Fin"] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today", format="YYYY-MM-DD")
    st.session_state.filters["A_Score"] = st.slider("Atypicité", min_value=0, max_value=10, value=0, step=1)

    st.session_state.filtered = st.form_submit_button(label="Enregistrer")
    if st.session_state.filtered :
        with st.sidebar.status("Selection des données...") as status:
            st.session_state.filtered_data = filter_data(data, st.session_state.filters)
            status.update(label='Données filtrées', state = "complete")
            st.text(st.session_state.filtered_data)

clusters = st.sidebar.toggle("Affichage groupé")

with col1:
    sub_col1, sub_col2 = st.columns(2)
    
    sub_col1.subheader("Carte des observations")
    if type(st.session_state.filtered_data) != type(None):
        st.session_state.filters["N"] = sub_col2.slider("Combien d'observations afficher ?", min_value=0, max_value=len(st.session_state.filtered_data), value=50, step=50)
    else :
        st.session_state.filters["N"] = 50
        
    st_data = make_map(st.session_state.filtered_data, 
                       GRENOBLE, 
                       st.session_state.filters["N"],
                       clusters)
    
with col2:
    st.subheader("Metadonnées")
    if type(st_data['last_object_clicked_popup']) == type(None):
        st.write("Veuillez cliquer sur une observation pour afficher les données associées")
    else : 
        id_obs = int(st_data['last_object_clicked_popup'])
        st.write(f"ID : {id_obs}")
        st.write(f"espèce : {data.at[id_obs, 'Nom flore']}")
        st.write(f"observateur : {data.at[id_obs, 'PrenomNom']}")
        st.write(f"coordonnées : ({data.at[id_obs, 'Latitude']}, {data.at[id_obs, 'Longitude']}")


# TODO : lier le tableau à la carte pour sélectionner un point
if type(st.session_state.filtered_data) != type(None):
    st.subheader(f"Données brutes (n = {len(st.session_state.filtered_data)})")
    select_row = st.dataframe(st.session_state.filtered_data.head(st.session_state.filters["N"]), 
                 hide_index=True, 
                 selection_mode="single-row",
                 on_select="rerun")
    st.session_state.filtered_data.iloc[select_row.selection["rows"][0]]
    # if type(st_data['last_object_clicked_popup']) != type(None):
        
else :
    st.subheader("Données brutes")





