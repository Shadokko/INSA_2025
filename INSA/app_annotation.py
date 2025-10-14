import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import branca.colormap as cm


@st.cache_data
def load_data(filename):
    chunk_size = 10_000
    chunks = []
    for chunk in pd.read_csv(filename, sep=";", usecols=["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore"],chunksize=chunk_size):
        chunks.append(chunk)
    data = pd.concat(chunks, axis=0)
    return data


def add_markers(df, m, n=0):
    if type(df) != type(None):
        if n==0:
            n = len(df)
        else:
            n = min(n, len(df))
            
        for i in range(n):
            # rank = df.iloc[[i]]["rank_ground_truth"]
            folium.Marker(
                location = df.iloc[[i]][['Latitude', 'Longitude']],
        	popup = df.iloc[[i]]["nom flore"]).add_to(m)#,
            # color = "red",
         #        tooltip = str(rank),#"cliquez pour afficher",
         #        icon = folium.Icon(color = "blue", 
         #                           icon = "leaf", 
         #                           prefix = "fa"),
         #        ).add_to(map_)


def filter_data(data, filters):
    filtered_data = data.copy()
    for parametre in filters:
        if filters[parametre] != "":
            filtered_data = data.loc[filtered_data[parametre]==filters[parametre]]
    return filtered_data


def make_map(f_data):
    map_ = folium.Map(location=GRENOBLE, zoom_start=8)
    add_markers(f_data, map_, 50)
    st_folium(map_)

#st.set_page_config(layout="wide")

DATA_PATH = "result_export.csv"
GRENOBLE = (45.193620, 5.720363)

st.title("Outil d'annotation")

with st.sidebar.status("Chargement des données...") as status:
    data = load_data(DATA_PATH)
    status.update(label='Données à jour', state = "complete")

if "filtered" not in st.session_state:
    st.session_state.filtered = False
    st.session_state.filtered_data = None

st.sidebar.subheader("Filtres")
with st.sidebar.form(key="filtres2"):
    st.session_state.filters = dict()
    st.session_state.filters["PrenomNom"] = st.text_input("Nom de l'observateur*")
    #st.session_state.filters[debut] = st.date_input("Du", value = "1990-01-01", min_value="1990-01-01", max_value="today")
    # st.session_state.filters[fin] = st.date_input("Jusqu'au", value = "today", min_value="1990-01-01", max_value="today")
    # st.session_state.filters[score] = st.select_slider("Atypicité", options=[1,2,3])
    
    st.session_state.filtered = st.form_submit_button(label="Enregistrer")
    if st.session_state.filtered :
        with st.sidebar.status("Selection des données...") as status:
            st.session_state.filtered_data = filter_data(data, st.session_state.filters)
            status.update(label='Données filtrées', state = "complete")
            st.text(st.session_state.filtered_data)
            

if st.session_state.filtered :
    if st.session_state.filters["PrenomNom"] == "":
        st.error("Veuillez choisir un observateur.")
    elif st.session_state.filters["PrenomNom"] not in list(data["PrenomNom"]):
        st.error(f"Impossible de trouver cet observateur ({st.session_state.filters["PrenomNom"]}). Veuillez vérifier l'orthographe er réessayer.")


#color_map = cm.LinearColormap(["green", "yellow", "red"],vmin=min(data["rank_ground_truth"]), vmax=max(data["rank_ground_truth"]))

make_map(st.session_state.filtered_data)


