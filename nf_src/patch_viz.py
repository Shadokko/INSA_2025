import pandas as pd
import numpy as np
import sys

from sklearn.model_selection import train_test_split
from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor

from pathlib2 import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('TkAgg')

# SETTINGS
# files
ROOT_PATH = Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm')
path2outputfolder = Path(r'C:\Users\fauren\PycharmProjects\bac_a_sable\GP\databases\output')

DATASET_PATH = ROOT_PATH / 'data/full_dataset.csv'
RASTER_PATH = ROOT_PATH / 'data/rasters_GLC19'
# csv columns
ID = 'id'
LABEL = 'Label'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'

# dataset construction
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# environmental patches
PATCH_SIZE = 64

# Patch location
lat, lng = 45.1935998, 5.7191127  # NF's home :-)


# examine dataset
df = pd.read_csv(DATASET_PATH, header='infer', sep=';', low_memory=False)

stats = np.unique(df.Species, return_counts=True)

print(f'Number of instances: {len(df)}')
print(f'Number of species: {stats[0].shape[0]}')

# ids = df[ID].to_numpy()
# labels = df[LABEL].to_numpy()
# positions = df[[LATITUDE, LONGITUDE]].to_numpy()
# splitting train val test
# train_labels, test_labels, train_positions, test_positions, train_ids, test_ids\
#     = train_test_split(labels, positions, ids, test_size=TEST_SIZE, random_state=42)
# train_labels, val_labels, train_positions, val_positions, train_ids, val_ids\
#     = train_test_split(train_labels, train_positions, train_ids, test_size=VAL_SIZE, random_state=42)

# create patch extractor
extractor = PatchExtractor(str(RASTER_PATH), size=PATCH_SIZE, verbose=True)
# add all default rasters
extractor.add_all()

# examining a patch

patch = extractor[(lat, lng)]
print(f'Patch shape: {patch.shape}')
print(f'Size of the patch in memory: {patch.size * patch.itemsize / 1024 / 1024:.3f} Mo')

plt.imshow(patch[0])

patch_2 = extractor.__getitem__((lat, lng), cancel_one_hot=True)  # with no one hot encoding


# List the variables in the extractor
var_list = [el.name for el in  extractor.rasters]
print(f'Nb of variables: {len(var_list)}\nVariables: {var_list}') # see p37 of Thesis for details


# images of the patch
def show_patch(patch,
               var_list=[el.name for el in  extractor.rasters],
               outputfolder=path2outputfolder,
               output_filename='patch64.png',
               patch_size=PATCH_SIZE):
    fig = plt.figure(figsize=(14, 12), dpi=150)
    for i, layer in enumerate(var_list):
        plt.subplot(4, 9, i + 1)
        plt.imshow(patch[i])
        plt.title(layer)

    plt.suptitle(f'Visualisation du patch {patch_size}x{patch_size} centré en {lat}N, {lng}E (Porte de France près de Grenoble)\n')
    plt.tight_layout()
    plt.show()
    plt.savefig(str(outputfolder / output_filename))
    return fig

show_patch(patch_2, output_filename=f"patch_{PATCH_SIZE}.png")

print("Done!")
sys.exist(0)