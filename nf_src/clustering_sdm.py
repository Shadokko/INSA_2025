import numpy as np
import pandas as pd
import sys
import umap

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pathlib2 import Path
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt

import re

import time
import hdbscan

import shapely
import geopandas as gpd
import json
import sys


tics = {}

# parameters
path2grid = Path(r'C:\Users\fauren\Documents_nosync\cnn_sdm_output\experiments\top1200\default_model\result_export_grid.csv')
path2if = Path(r'C:\Users\fauren\Documents_nosync\bac_a_sable_GP_databases\InFloris\infloris_reformat.csv')
path2outputfolder = Path(r'C:\Users\fauren\Documents_nosync\bac_a_sable_GP_databases\output\oct2023')

n_components = 3
grid_sample = 5000

# loading
tics['load'] = time.time()
print('Loading data: ')
grid_pred = pd.read_csv(path2grid, sep=';')
infloris = pd.read_csv(path2if, sep=';')

print(f"Time to load data: {time.time() - tics['load']:.2f} seconds")

# computing relative frequencies (predicted presence probability with respect to mean predicted presence probability)
print('generating relative frequencies table...')
relative_frequencies = grid_pred.iloc[:, -1200:].copy(deep=True)
for column in relative_frequencies.columns:
    relative_frequencies.loc[:, column] = relative_frequencies.loc[:,column] / np.mean(relative_frequencies.loc[:,column])



print('generating absolute probability table...')
abs_probas = grid_pred.iloc[:, -1200:].copy(deep=True)

## get geometry
infloris['geometry'] = gpd.points_from_xy(x=infloris["Longitude"], y=infloris["Latitude"], crs="EPSG:4326")
infloris_geom = gpd.GeoDataFrame(infloris)

## get contour
contour = gpd.GeoSeries(shapely.buffer(infloris_geom.sample(10000, random_state=42).geometry, 0.1).unary_union.simplify(tolerance=0.02))

## rename most probable columns
new_cols = [col if '/100' not in col else f'Taxon probable {col[:-4]}' for col in grid_pred.columns]
grid_pred.columns = new_cols


# k-means clustering on relative frequencies
# clustering
print("k-means clustering")
for n_cluster in [5, 10, 20, 30, 50, 100, 200]:
    tics[f'km_{n_cluster}'] = time.time()
    km_clusterer = KMeans(n_clusters=n_cluster, n_init=10, random_state=42)
    km_labels = km_clusterer.fit_predict(relative_frequencies)
    grid_pred[f'km_cluster_{n_cluster}'] = km_labels # todo: add purity
    print(f"Time to fit and predict kmeans for {n_cluster}: {time.time() - tics[f'km_{n_cluster}']:.2f} seconds")

    # adding distances from cluster centroids
    transform = km_clusterer.transform(relative_frequencies)
    for n in range(n_cluster):
        grid_pred[f'dist_from_centroid_cluster_{n}_{n_cluster}'] = transform[:, n]
    grid_pred[f'Distance au centroïde le plus proche (sur {n_cluster})'] = np.apply_along_axis(np.min, 1, transform)

    # A transformation essentially used for display purposes
#    grid_pred[f'closeness_to_closest_centroid_{n_cluster}'] = (5 * (1 - (
#                grid_pred[f'dist_from_closest_centroid_{n_cluster}'] /
#                np.max(grid_pred[f'dist_from_closest_centroid_{n_cluster}'])) ** 0.2)).astype('int')

    print(f"Analyzing cluster")
    # todo: elaborate. get transormed data and adjust size/opacity depending on distance to the centroid.
    # Get basic statistics (most frequent), visualization of most probable/ most frequent species
    # Give an index of specificify / originality for each cluster

    print("Giving a name to the cluster, corresponding to the highest abs and relative frequencies")
    grid_pred[f'km_cluster_{n_cluster}_name'] = None

    names = []
    for cluster in range(n_cluster):
        cluster_abs = abs_probas.loc[grid_pred[f'km_cluster_{n_cluster}'] == cluster, :]
        name_abs = cluster_abs.columns[np.argmax(np.mean(cluster_abs))]

        cluster_freq = relative_frequencies.loc[grid_pred[f'km_cluster_{n_cluster}'] == cluster, :]
        name_freq = cluster_freq.columns[np.argmax(np.mean(cluster_freq))]

        name = re.split(' ', name_abs)[0] + '_' + re.split(' ', name_abs)[1] \
               + '__' + re.split(' ', name_freq)[0] + '_' + re.split(' ', name_freq)[1]

        if name in names:  # dealing with dedundant names
            name += f'_{names.count(name) + 1}'
        names.append(name)

        print(f'Name for cluster {cluster}: {name}')
        grid_pred.loc[grid_pred[f'km_cluster_{n_cluster}'] == cluster, f'km_cluster_{n_cluster}_name'] = name


    print(f'Generating map for {n_cluster} clusters')
    # todo: overlay actual observations, change size/opacity according to typicality versus cluster
    fig1 = px.scatter_mapbox(grid_pred, lat='Latitude', lon='Longitude',
                             color=f'km_cluster_{n_cluster}_name',
                             opacity=0.4,
                             size_max=1,
                             hover_data=[f'Distance au centroïde le plus proche (sur {n_cluster})'] + 
                                        [f'Taxon probable {n}' for n in range(1, 6)])
    fig1.update_layout(
        mapbox={
            "style": "open-street-map",
            "layers": [
                {
                    "source": json.loads(contour.to_json()),
                    "below": "traces",
                    "type": "line",
                    "color": "purple",
                    "line": {"width": 3},
                }
            ],
        })
    # fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig1.update_layout(title_text=f"Distribution potentielle pour un nombre de cluster de {n_cluster}"
                                  f"<br>Les noms de communautés végétales correspondent à l'espèce la plus probable, suivi de l'espèce la plus caractéristique")
    fig1.write_html(str(path2outputfolder / f'cluster_map_{n_cluster}.html'))
    fig1.show()

# saving
grid_pred.to_csv(path2outputfolder / 'grid_pred_cluster.csv', sep=';', index=False)

print('Done !') # what follows is archive...
sys.exit(0)




print(f'Subsetting grid to a sample of {grid_sample}')
grid_small = grid_pred.sample(grid_sample, random_state=42)




for n_cluster in [10, 30, 100]:
    tics[f'km_{n_cluster}'] = time.time()
    km_clusterer = KMeans(n_clusters=n_cluster, init='random', n_init=5)
    km_labels = km_clusterer.fit_predict(grid_small.iloc[:, -1200:])
    grid_small[f'km_cluster_{n_cluster}'] = km_labels
    print(f"Time to fit and predict kmeans for {n_cluster} clusters: {time.time() - tics[f'km_{n_cluster}']:.2f} seconds")



print('Dimensionality reduction')
# PCA reduction
tics['pca'] = time.time()
pca = PCA(n_components=n_components)
pca_embedding = pca.fit_transform(grid_small.iloc[:,-1200:].values)
print(f"Time to compute PCA projection: {time.time() - tics['pca']:.2f} seconds")

# UMAP reduction
tics['umap'] = time.time()
umap_reducer = umap.UMAP(n_components=n_components)
umap_embedding = umap_reducer.fit_transform(grid_small.iloc[:,-1200:].values)
print(f"Time to compute UMAP projection: {time.time() - tics['umap']:.2f} seconds")


# visualization
tics['visu'] = time.time()
for n in range(n_components):
    grid_small[f'pca_{n + 1}'] = pca_embedding[:, n]
    grid_small[f'umap_{n + 1}'] = umap_embedding[:, n]

fig_umap = px.scatter(grid_small, x="umap_1", y="umap_2", color="1/100",
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_umap.update_layout(title_text=f"Projection UMAP des données générée sur un échantillon de {grid_sample}")
fig_umap.write_html(str(path2outputfolder / ('umap.html')))
fig_umap.show()

fig_pca = px.scatter(grid_small, x="pca_1", y="pca_2", color="1/100",
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_pca.update_layout(title_text=f"Projection PCA des données générée sur un échantillon de {grid_sample}")
fig_pca.write_html(str(path2outputfolder / ('pca.html')))
fig_pca.show()

fig_pca123 = px.scatter_3d(grid_small, x="pca_1", y="pca_2", z='pca_3', color="1/100",
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_pca123.update_layout(title_text=f"Projection PCA des données générée sur un échantillon de {grid_sample}")
fig_pca123.write_html(str(path2outputfolder / ('pca_3D.html')))
fig_pca123.show()


fig_umap123 = px.scatter_3d(grid_small, x="umap_1", y="umap_2", z="umap_3", color="1/100",
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_umap123.update_layout(title_text=f"Projection UMAP des données générée sur un échantillon de {grid_sample}")
fig_umap123.write_html(str(path2outputfolder / ('umap_3D.html')))
fig_umap123.show()

print(f"Time to generate visualization: {time.time() - tics['visu']:.2f} seconds")

# projecting on the full grid dataset
tics['apply_umap'] = time.time()
embedding_umap_applied = umap_reducer.transform(grid_pred.iloc[:,-1200:].values)
print(f'Time to apply UMAP to the full grid dataset: {time.time() - tics["apply_umap"]:.2f} seconds')

for n in range(n_components):
    grid_pred[f'umap_{n + 1}'] = embedding_umap_applied[:, n]

fig_umap123_full = px.scatter_3d(grid_pred, x="umap_1", y="umap_2", z="umap_3", color="1/100",
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_umap123_full.update_layout(title_text=f"Projection UMAP des données générée sur un échantillon de {grid_sample}, et appliqué sur tous le jeu de données 'grid'")
fig_umap123_full.write_html(str(path2outputfolder / ('umap_3D_full.html')))
fig_umap123_full.show()



for min_cluster_size in [100, 200, 500]:
    tics['clustering'] = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(grid_pred.loc[:, ['umap_1', 'umap_2', 'umap_3']])
    print(f'Time to apply HDBSCAN clustering with min clustering size {min_cluster_size}, on the full grid dataset: {time.time() - tics["clustering"]:.2f} seconds')

    grid_pred[f'cluster_{min_cluster_size}'] = clusterer.labels_
    grid_pred[f'cluster_{min_cluster_size}_proba'] = clusterer.probabilities_

# Some visualizations
clusterer.condensed_tree_.plot()
plt.show()

fig_umap123_clust = px.scatter_3d(grid_pred.sample(10000), x="umap_1", y="umap_2", z="umap_3", color='cluster_500',
                 hover_data=['1/100', '2/100', '3/100', '4/100', '5/100'])
fig_umap123_clust.update_layout(title_text=f"Projection UMAP des données générée sur un échantillon de {grid_sample}, et appliqué sur tous le jeu de données 'grid'")
fig_umap123_clust.write_html(str(path2outputfolder / ('umap_3D_full_cluster.html')))
fig_umap123_clust.show()

for min_cluster_size in [100, 200, 500]:
    n_cluster = np.max(grid_pred[f'cluster_{min_cluster_size}'])
    for cluster in range(n_cluster):
        print(f'Processing: {cluster} / {n_cluster}')
        # cluster = 'Arenaria multicaulis L., 1759'
        # fig1 = px.scatter_mapbox(grid_pred, color=cluster, opacity=cluster, lat="Latitude", lon="Longitude", zoom=8)
        fig1 = px.density_mapbox(grid_pred.loc[grid_pred['cluster_500'] == cluster, :],
                                 lat='Latitude', lon='Longitude', z='cluster_500_proba', hover_data=[f'{n}/100' for n in range(1,21)],
                                 radius=5)
        fig1.update_layout(mapbox_style="open-street-map")
        # fig1.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig1.update_layout(title_text=f"Distribution potentielle pour cluster  {cluster}")
        fig1.write_html(str(path2outputfolder / (f'cluster_map_minsize{min_cluster_size}_{cluster}.html')))
        # fig1.show()
    #
# saving
grid_pred.to_csv(path2outputfolder / 'grid_pred_umap.csv', sep=';', index=False)




print('Done !')
sys.exit(0)