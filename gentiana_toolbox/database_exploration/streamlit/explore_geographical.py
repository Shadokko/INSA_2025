"""
A streamlit app to explore geographical data, such as botanical observations

usage:
- go to root folder
- on the command line (in linux):
python3 -m streamlit run ./database_exploration/streamlit/explore_geographical.py
(in windows)
.\.venv\Scripts\python.exe -m streamlit run .\gentiana_toolbox\database_exploration\streamlit\explore_geographical.py

"""

# TODO: add a second, dynamic map, to select areas
# TODO: drag and drop new files
# TODO: inline formatting (adding ecological elements, address, altitude, etc...)
# TODO: display stats (species frequency, observer frequency, etc.), date
# TODO: merge with other databases (Family, ecological groups, etc.)
# TODO: retrieve commune names from insee codes

import streamlit as st
import pandas as pd
from pathlib2 import Path
# import yaml
from ruamel.yaml import YAML
import folium
from folium import CircleMarker
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.io as pio
pio.templates.default = "ggplot2"

import random
import numpy as np

# Reduce main map width and sidebar width for a more compact display
st.set_page_config(layout="wide")

__path2parameter_default__ = Path('./gentiana_toolbox/database_exploration/streamlit/parameters.yml') # Path('./gentiana_toolbox/database_exploration/streamlit/parameters.yml')

__palette__ = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'white', 'lightred','lightblue', 'black']

__longer_palette__ = ['#FF6347', '#4682B4', "#9E9D9A", '#FF69B4', '#00CED1', '#8A2BE2', '#7FFF00', 
                      'black', 'lightgrey', '#FF8C00', '#8B0000', '#FF00FF', '#FFDAB9',  '#6A5ACD', '#7CFC00', '#00FA9A', 
                     '#20B2AA', '#FF7F50', '#778899', '#B0E0E6', '#FFA07A', '#B22222', '#87CEFA', '#FF4500',
                     '#DAA520', '#98FB98', '#DDA0DD', '#F08080', '#4682B4', '#D2691E', '#32CD32']

palette = __longer_palette__  #__palette__

# @st.cache_data
def load_parameters(path2file=__path2parameter_default__):
    if not Path(path2file).exists():
        return {}
    yaml = YAML(typ='safe')
    return yaml.load(Path(path2file))
    # with open(Path(path2file), 'r') as file:
    #     return yaml.load(file, Loader=yaml.FullLoader)

path2parameters = Path(__path2parameter_default__)
params = load_parameters(path2parameters)

# get parameters
col_species = params['colonnes']['espece']
col_date = params['colonnes']['date']
col_year = params['colonnes']['annee']
col_month = params['colonnes']['mois']
col_observer = params['colonnes']['observateur']
col_freq_all = params['colonnes']['frequence']
col_altitude = params['colonnes']['altitude']
col_latitude = params['colonnes']['latitude']
col_longitude = params['colonnes']['longitude']

col_ville = params['colonnes']['ville']
default_ville = params['valeurs_a_defaut']['ville']

tile = params['geoportail']['tile']
attr = params['geoportail']['attr']

verbose = params['verbose']

if verbose:
    print(f"Latitude column: {col_latitude}")
    print(f"longitude column: {col_longitude}")

@st.cache_data
def load_obs(path2file=params['input']['path2obs'], verbose=True):
    if verbose: print(f"Dropping lines with no species, date, year, month")
    df = pd.read_csv(Path(path2file), sep=';')
    if verbose: print(f'Initial length of DataFrame: {len(df)}')
    df = df.dropna(subset=[col_species, col_date, col_year, col_month])
    if verbose: print(f'Length of DataFrame after dropping: {len(df)}')
    return df

@st.cache_data
def get_list(df, column=col_species):
    return sorted(df[column].unique().astype(str))

@st.cache_data
def get_extreme(df, column=col_year):
    return [df[column].min(), df[column].max()]

@st.cache_data
def get_frequency(df, column=col_species):
    """
    Get the frequency of each value in the specified column.
    """
    if column not in df.columns:
        st.error(f"Column '{column}' not found in the dataframe.")
        return pd.Series(dtype='int')
    else:
        return df[column].value_counts()

@st.cache_data
def filter_year_month(df, year_min, year_max, month_min, month_max):
    return df.loc[(df.Annee >= year_min) & (df.Annee <= year_max) & (df[col_month] >= month_min) & (df[col_month] <= month_max), :]

@st.cache_data
def filter_freq_species(df, freq_min, freq_max):
    return df.loc[(df[col_freq_all] >= freq_min) & (df[col_freq_all] <= freq_max), :]

@st.cache_data
def filter(df, col, col_val):
    return df.loc[df_sub[col].astype(str).isin(col_val), :]

@st.cache_data
def filter_by_content(df, col, col_val):
    if ',' in col_val:
        col_val = col_val.split(',')
    elif ';' in col_val:
        col_val = col_val.split(';')
    else: col_val = [col_val]
    # remove ' ' at beginning and end of each value
    col_val = [val.strip() for val in col_val]
    indexes = [i for i, val in enumerate(df[col].astype(str)) if any(c in val for c in col_val)]
    return df.iloc[indexes, :]   

@st.cache_data
def filter_numerical(df, num_min, num_max, col_numerical=col_altitude):
    """
    Filter the dataframe based on altitude range.
    """
    return df.loc[(df[col_numerical] >= num_min) & (df[col_numerical] <= num_max), :]


@st.cache_data
def copy_dataset(df):
    return df.copy()

@st.cache_data
def blur_lat_lon(df, blur_factor=0.0001):
    """
    Apply a random blur to the latitude and longitude columns of the dataframe.
    This is useful to avoid overlapping markers on the map.
    """
    df['Latitude'] += np.random.uniform(-blur_factor, blur_factor, size=len(df))
    df['Longitude'] += np.random.uniform(-blur_factor, blur_factor, size=len(df))
    return df

@st.cache_data
def sort_dataframe(df, col=col_species):
    """
    Sort the dataframe by the specified column frequency.
    """
    if col not in df.columns:
        st.error(f"Column '{col}' not found in the dataframe.")
        return df, []
    else:
        #sorting df by most frequent values in the specified column
        df = df.sort_values(by=col, key=lambda x: x.map(x.value_counts()), ascending=False)
        return df, df.value_counts(subset=col, sort=True)

@st.cache_data
def filter_frequency(df, column, rank_freq_min=1, rank_freq_max=100, verbose=True):
    """
    Filter the dataframe based of the frequency of col instances.
    """
    freq_table = df[column].value_counts()
    freq_table = freq_table.sort_values(ascending=False)
    valid_values = freq_table[rank_freq_min - 1: rank_freq_max].index
    if verbose:
        print(f"Filtering {column} with frequency ranks between {rank_freq_min} and {rank_freq_max}.")
        # print(f"corresponding Nb of occurrences range. Max: {freq_table[rank_freq_min - 1]}, Min: {freq_table[rank_freq_max - 1]}")
        print(f"Valid values: {valid_values.tolist()}")
        print(f"Number of valid values: {len(valid_values)}")
    return df[df[column].isin(valid_values)]   

@st.cache_data
def add_int_column(df, column=col_altitude, verbose=True):
    """
    Add an integer column to the dataframe based on the specified column.
    This is useful to avoid issues with float values in the altitude column.
    """
                      # Drop rows with NaN altitude
    if column not in df.columns:
        st.error(f"Column '{column}' not found in the dataframe.")
        return df
    else:
        if verbose:
            print(f"Adding integer column '{column}_int' based on '{column}'")
            length = len(df)

        df.dropna(subset=[column], inplace=True)
        if verbose:
            print(f"Number of dropped NaN values in {column}: {len(df) - length}")
            print(f"Number of remaining rows: {len(df)}")
        df[f'{column}_int'] = df[column].astype(int)
        return df

@st.cache_data
def get_map_display(df_sub):
    # TODO: set default zoom and center based on the data
    lat_mean = df_sub['Latitude'].mean() if 'Latitude' in df_sub.columns else 45.2
    lon_mean = df_sub['Longitude'].mean() if 'Longitude' in df_sub.columns else 5.8

    lat_min, lat_max = get_extreme(df_sub, column='Latitude')
    lon_min, lon_max = get_extreme(df_sub, column='Longitude')
    max_range = max(lat_max - lat_min, lon_max - lon_min) 

    if max_range > 1: zoom_start = 8
    elif max_range > 0.5: zoom_start = 9
    elif max_range > 0.2: zoom_start = 10
    elif max_range > 0.07: zoom_start = 11
    else: zoom_start = 12

    return lat_mean, lon_mean, zoom_start

# @st.cache_data
def add_markers(df, color_freq_table, col_color, 
                max_markers=5000, mode =  ['groups', 'marker_cluster', 'heat_map']):
    if len(df) > max_markers:
        df = df.sample(max_markers)
    
    # color_values = df[col_color].unique()
    df['color'] = 'black'

    groups = {}
    for index, color_value in enumerate(color_freq_table.index):
        df.loc[df[col_color] == color_value, 'color'] = palette[index % len(palette)]
        groups[color_value] = folium.FeatureGroup(name=color_value)

    marker_cluster = MarkerCluster(name='Clusters')
    heat_map = []
    for i, row in df.iterrows():
        if 'groups' in mode:
            marker = CircleMarker([row.Latitude, row.Longitude], radius=5, fill=True, color=row['color'], 
                                  tooltip=f'{row[col_species]} / {row[col_latitude]:.2f} {row[col_longitude]:.2f}', 
                                  popup= f'{row[col_observer]} - {row[col_date]} - {row[col_latitude]} - {row[col_longitude]}')
            marker.add_to(groups[row[col_color]])
        if 'marker_cluster' in mode:
            CircleMarker([row.Latitude, row.Longitude], radius=5, fill=True, color=row['color'], 
                         tooltip=f'{row[col_species]} - {row[col_latitude]:.2f} - {row[col_longitude]:.2f}', 
                         popup= f'{row[col_observer]} - {row[col_date]} / {row[col_latitude]:.4f}  {row[col_longitude]:.4f}').add_to(marker_cluster)
        if 'heat_map' in mode:
            heat_map.append([row.Latitude, row.Longitude, 1])
    return groups, marker_cluster, heat_map


@st.cache_data
def generate_folium_map(df_sub, color_freq_table, col_color, 
                        lat_mean, lon_mean, zoom_start, tile=tile, attr=attr, 
                        display_clusters=False, display_heatmap=False, max_num=5000):
    m = folium.Map([lat_mean, lon_mean], zoom_start=zoom_start, tiles=tile, attr=attr)

    if display_clusters&display_heatmap:
        _, marker_cluster, heat_map = add_markers(df_sub, color_freq_table=color_freq_table, col_color=col_color, 
                                        max_markers=max_num, mode = ['marker_cluster', 'heat_map'])
        HeatMap(heat_map, name='Densité').add_to(m)
        marker_cluster.add_to(m)
    elif display_heatmap:
        groups, _, heat_map = add_markers(df_sub, color_freq_table=color_freq_table, col_color=col_color, 
                                max_markers=max_num, mode = ['groups', 'heat_map'])
        HeatMap(heat_map, name='Densité').add_to(m)
        for group in groups.values():
            group.add_to(m)
    elif display_clusters:
        groups, marker_cluster, _ = add_markers(df_sub, color_freq_table=color_freq_table, col_color=col_color, 
                                            max_markers=max_num, mode = ['groups', 'marker_cluster'])
        marker_cluster.add_to(m)
        for group in groups.values():
            group.add_to(m)
    else:
        groups, _, _ = add_markers(df_sub, color_freq_table=color_freq_table, col_color=col_color, 
                                max_markers=max_num, mode = ['groups'])
        for group in groups.values():
            group.add_to(m)

    folium.LayerControl().add_to(m)
    return m

@st.cache_data
def generate_histogram(df_sub, min_alt, max_alt, bin_width=250, 
                  column=col_altitude, col_color=col_species, 
                  max_facet=20,
                  verbose=True):
    """
    Generate a histogram of the specified column in the dataframe.

    Parameters:
    - df_sub: DataFrame containing the data

    - column: Column to generate the histogram for (default is 'altitude_int')
    - bin_width: Width of each bin (default is 250)
    - col_color: Column to color the histogram by (default is 'espece')
    Returns:

    """
    min_alt = (min_alt // bin_width) * bin_width  # Round down to nearest bin width
    max_alt = ((max_alt // bin_width) + 1) * bin_width  # Round up to nearest bin width
    nbins = int((max_alt - min_alt) // bin_width)

    if verbose:
        print(f"Generating histogram for {column} with bins from {min_alt} to {max_alt} (bin width: {bin_width})")
        print(f"Number of bins: {nbins}")
        print(f"Coloring by: {col_color}")

    # limit to the 20 most frequent values of col_color
    if df_sub[col_color].nunique() <= max_facet:
        n_facets = df_sub[col_color].nunique()
        n_rows = (n_facets // 5) + 1
        # bin_edges = np.arange(min_alt, max_alt + bin_width, bin_width)
        # df_sub['alt_bin'] = pd.cut(df_sub['altitude_int'], bins=bin_edges, include_lowest=True)
        # bin_counts = df_sub['alt_bin'].value_counts().sort_index()

        # hist = px.bar(
        #     x=bin_counts.values,
        #     y=[str(interval) for interval in bin_counts.index],
        #     orientation="h",
        #     labels={'x': 'Count', 'y': 'Altitude (bin)'},
        #     title='Histogramme horizontal des altitudes (binning manuel)', 
        #     facet_col=col_color, color=col_color,
        #     color_discrete_sequence=palette,    
        #     facet_col_wrap=5, height=300 * n_rows + 200,)

        hist = px.histogram(
            df_sub,
            y='altitude_int',
            facet_col=col_color,
            facet_col_wrap=5,
            title=f'Histogrammes des altitudes, par {col_color}',
            labels={'altitude_int': 'Altitude (m)'},
            nbins=nbins,
            range_y=[min_alt, max_alt],
            orientation="h",
            height=300 * n_rows + 200, 
        )
        for i, trace in enumerate(hist.data):
            trace.marker.color = palette[i % len(palette)]
        # Tilt facet subtitles (annotations)
        for annotation in hist.layout.annotations:
            if annotation.text.startswith(f"{col_color}="):
                annotation.text = annotation.text.replace(f"{col_color}=", "")
            annotation.textangle = -5  # or any angle you prefer, e.g., -30

    else:
        hist = px.histogram(
            df_sub,
            title='Histogramme des altitudes (global)',
            labels={'altitude_int': 'Altitude (m)'},
            nbins=nbins,
            range_y=[min_alt, max_alt],
            height= 500
        )
    return hist


df = load_obs(params['input']['path2obs'], verbose=True)
df = add_int_column(df, column=col_altitude)  # Add an integer column for altitude
# df_sub = pd.DataFrame(columns=df.columns)  # Initialize an empty DataFrame with the same columns

# df = uploaded_file if uploaded_file else load_obs()

extreme_alt = get_extreme(df, column=f"{col_altitude}_int")
extreme_freq = get_extreme(df, column=col_freq_all)
extreme_year = get_extreme(df, column=col_year)
extreme_lat = get_extreme(df, column=col_latitude)
extreme_lon = get_extreme(df, column=col_longitude)

col_color = col_species  # Default color column is the species column

st.title('Exploration de données géographiques')

# modify the following to make sure the sidebar is split into two columns
col_side, col_display, col_graph = st.columns([3, 2, 10])
with col_side:
    with st.form("Filters"):
        st.subheader('Formulaire de filtrage')
        
        col_sideleft_0, col_sideright_0 = st.columns([2,1])
        with col_sideleft_0:
            st.text('Appuyer pour générer les graphes')
        with col_sideright_0:
            submitted_filters = st.form_submit_button("Filtrer")
        
        st.caption('----------------------------------')
        col_sideleft, col_sideright = st.columns(2)

        with col_sideleft:
            lat_min, lat_max = st.slider('Latitude', extreme_lat[0], extreme_lat[1], extreme_lat)
            year_min, year_max = st.slider('Période', extreme_year[0], extreme_year[1], extreme_year)
            freq_min, freq_max = st.slider('Fréquence espèce', extreme_freq[0], extreme_freq[1], 
                                            extreme_freq)
            col1 = st.selectbox('Filtre 1 sur: ', df.columns, index=df.columns.get_loc(col_species))
            col1_val = st.text_input(f"{col1} contient:", value='')  # text input for observer name
            col3 = st.selectbox('Filtre 3 sur: ', df.columns, index=df.columns.get_loc(col_ville)) #TODO: filter on col1 first
            col3_val = st.multiselect(f"Valeur(s) de {col3}:", get_list(df, col3), default=None)
        with col_sideright:
            lon_min, lon_max = st.slider('Longitude', extreme_lon[0], extreme_lon[1], extreme_lon)            
            month_min, month_max = col_sideright.slider('Mois', 1, 12, [1, 12])
            alt_min, alt_max = col_sideright.slider('Altitude', extreme_alt[0], extreme_alt[1], extreme_alt)
            col2 = col_sideright.selectbox('Filtre 2 sur: ', df.columns, index=df.columns.get_loc(col_observer))
            col2_val = col_sideright.text_input(f"{col2} contient:", value='')  # text input for observer name
            col4 = st.selectbox('Filtre 4 sur: ', df.columns, index=df.columns.get_loc(col_species)) #TODO: filter on col2 first
            col4_val = st.multiselect(f"Valeur(s) de {col4}:", get_list(df, col4), default=None)

        col5 = st.selectbox('Filtre 5 (fréquence) sur: ', df.columns, index=df.columns.get_loc(col_species))
        col5_freq = get_frequency(df, column=col5)
        extreme_col5 = [1, len(col5_freq)]
        col5_val = st.slider(f"Fréquence de {col5}", extreme_col5[0], extreme_col5[1], extreme_col5)


    # If the form is submitted, filter the dataframe based on the selected values
    if submitted_filters:
        df_sub = copy_dataset(df) # pd.DataFrame(columns=df.columns)
        if [lat_min, lat_max] != extreme_lat: df_sub = filter_numerical(df_sub, lat_min, lat_max, col_numerical=col_latitude)
        if [lon_min, lon_max] != extreme_lon: df_sub = filter_numerical(df_sub, lon_min, lon_max, col_numerical=col_longitude)
        if col3_val != []: df_sub = filter(df_sub, col3, col3_val)
        if col4_val != []: df_sub = filter(df_sub, col4, col4_val)
        if col1_val != '': df_sub = filter_by_content(df_sub, col1, col1_val)
        if col2_val != '': df_sub = filter_by_content(df_sub, col2, col2_val)
        if [freq_min, freq_max] != extreme_freq: df_sub = filter_freq_species(df_sub, freq_min, freq_max)
        if [year_min, year_max] != extreme_year or [month_min, month_max] != [1, 12]:
            df_sub = filter_year_month(df_sub, year_min, year_max, month_min, month_max)
        if [alt_min, alt_max] != extreme_alt: df_sub = filter_numerical(df_sub, alt_min, alt_max)

        if col5_val != extreme_col5: df_sub = filter_frequency(df_sub, column=col5, rank_freq_min=col5_val[0],  rank_freq_max=col5_val[-1])
        st.session_state['df_sub'] = df_sub
    elif 'df_sub' in st.session_state:
        df_sub = st.session_state['df_sub']
    else:
        df_sub = pd.DataFrame(columns=df.columns)

with col_display:
    with st.form("Display"):
        st.subheader('Formulaire d\'affichage')
        submitted_display = st.form_submit_button("Afficher")
        col_color = st.selectbox('Colorer les marqueurs sur: ', df.columns, index=df.columns.get_loc(col_species))
        # subcol_display_1, subcol_display_2 = st.columns(2) 
        # with subcol_display_1:
        st.caption('Carte')
        display_map = st.checkbox('Afficher la carte', value=True)
        max_num = st.number_input('Nombre maximum de marqueurs sur carte', value=2000)
        nb_legend = st.number_input(f"N maximum de {col_color} dans la légende", value=20, min_value=1, max_value=100)
        blur_factor = st.number_input('Facteur de flou pour les marqueurs', value=0.00005, step=0.00001, format="%.5f")
        display_clusters = st.checkbox('Afficher sous forme de clusters', value=False)
        display_heatmap = st.checkbox('Afficher la carte de chaleur', value=False)
        # with subcol_display_2:
        st.caption('Histogramme')
        display_hist = st.checkbox('Afficher l\'histogramme des altitudes', value=False)
        

    if submitted_display:
        st.session_state['col_color'] = col_color

if submitted_filters or submitted_display:

    with col_display:
        # st.caption('----------------------------------')
        export_format = st.selectbox('Exporter au format:', ['CSV', 'Excel'])
        if export_format == 'CSV':
            st.download_button(
                label="Télécharger le fichier CSV",
                data=df_sub.to_csv(index=False, sep=';').encode('utf-8'),
                file_name=Path(params['output']['path2export']).name,
                mime='text/csv'
        )
        elif export_format == 'Excel':
            st.download_button(
                label="Télécharger le fichier Excel",
                data=df_sub.to_excel(index=False, engine='openpyxl'),
                file_name=Path(params['output']['path2export']).name,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

    with col_graph:
        # add an input to select the file, or drag and drop
        
        if len(df_sub) == 0:
            st.text('En attente de données filtrées...')
            st.caption(f'    _Parameters: {params}_')
            st.caption(f'    _Path to root folder: {str(Path(".").resolve())}_')
            st.caption(f'    _Path to parameters file: {str(path2parameters.resolve())}_')
        else:
            df_sub = blur_lat_lon(df_sub, blur_factor=blur_factor)

            # df_sub, species_freq = sort_dataframe(df_sub, col=col_species)
            df_sub, color_freq_table = sort_dataframe(df_sub, col=col_color)
            
            if display_map:
                lat_mean, lon_mean, zoom_start = get_map_display(df_sub)
                
                col_legende, col_carte = st.columns([1, 3])
                
                with col_legende:
                    # Adding a legend to the figure, with the colors of the markers
                    # Génération dynamique de la légende
                    legend_items = ""
                    for idx, (color_name, color_freq) in enumerate(zip(color_freq_table.index, color_freq_table.values)):  # groups.keys()):
                        color = palette[idx % len(palette)]
                        legend_items += f'<p style="margin:0;"><span style="display:inline-block;width:12px;height:12px;background:{color};border-radius:50%;margin-right:8px;"></span>{color_freq} {color_name}</p>'
                        if idx >= nb_legend: # exit the loop if we have enough items
                            legend_items += '<p style="margin:0;"><span style="display:inline-block;width:12px;height:12px;background:grey;border-radius:50%;margin-right:8px;"></span>...</p>'
                            break

                    legend_html = f"""
                    <div style="width: 200px;
                        border:2px solid grey;
                        font-size:10px;
                        background-color: white;
                        padding: 5px;">
                    <p style="text-align: center; font-weight:bold;">Légende</p>
                    {legend_items}
                    </div>
                    """

                    st.markdown(legend_html, unsafe_allow_html=True)
                    text = f'Nombre d\'observations sélectionnées: {len(df_sub)}'
                    if len(df_sub) > max_num:
                        text += f"  (affichage d'un échantillon de {max_num} observations)"

                    st.write(text)
                    # st.markdown(f"<b>Nombre d'observations sélectionnées :</b> {len(df_sub)}", unsafe_allow_html=True)
                    st.markdown(f"<b>Nombre d'espèces :</b> {df_sub[col_species].nunique()}", unsafe_allow_html=True)
                    st.markdown(f"<b>Nombre d'observateurs :</b> {df_sub[col_observer].nunique()}", unsafe_allow_html=True)


                with col_carte:
                    map = generate_folium_map(df_sub, color_freq_table=color_freq_table, col_color=col_color,
                            lat_mean=lat_mean, lon_mean=lon_mean, zoom_start=zoom_start, 
                            tile=tile, attr=attr, display_clusters=display_clusters, 
                            display_heatmap=display_heatmap, max_num=max_num)
                    map_data = folium_static(map, width=700)




            if display_hist:
                st.subheader('Histogramme des altitudes')
                [min_alt, max_alt] = get_extreme(df_sub, column='altitude_int')
                hist = generate_histogram(df_sub, min_alt=100, max_alt=max(max_alt, 2100), 
                                        bin_width=250, 
                                        column='altitude_int', col_color=col_color, 
                                        max_facet=20, verbose=True)

                st.plotly_chart(hist, use_container_width=True)


