import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple

from pathlib2 import Path
# This is an example code how to use a scikit-learn model with our dataset (here a random forest classifier)

# SETTINGS
# files
TRAINSET_PATH = './data/train_dataset.csv'
TESTSET_PATH = './data/test_dataset.csv'
RASTER_PATH = './data/rasters_GLC19/'
# csv columns
ID = 'id'
LABEL = 'Label'
LATITUDE = 'Latitude'
LONGITUDE = 'Longitude'

# dataset construction
VAL_SIZE = 0.1

# environmental patches
PATCH_SIZE = 1

# model params
N_LABELS = 4520
MAX_DEPTH = 12
N_TREES = 100

# evaluation
METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30, 100]), ValidationAccuracyMultiple([1, 10, 30, 100]))

# create patch extractor
extractor = PatchExtractor(RASTER_PATH, size=PATCH_SIZE, verbose=True)
# add all default rasters
extractor.add_all()

# FINAL EVALUATION ON TEST SET
# read test set
df = pd.read_csv(TESTSET_PATH, header='infer', sep=';', low_memory=False)
ids = df[ID].to_numpy()
labels = df[LABEL].to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()
test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)

# load model
path = 'rf.skl'
print('Loading SKL model: ' + path)
clf = joblib.load(path)

# predict
inputs, labels = test_set.numpy()
restricted_predictions = clf.predict_proba(inputs)
predictions = np.zeros((restricted_predictions.shape[0], N_LABELS))
predictions[:, clf.classes_] = restricted_predictions

# convert predictions to top 100
path2sp_label = Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm\data\species_label_table.csv')
sp_label = pd.read_csv(path2sp_label, sep=';')

top100 = np.argsort(-predictions, axis=1)[:,:100]
top100_sp = np.empty(top100.shape, dtype=object)
truth_sp = np.empty(labels.shape, dtype=object)

for species, label in zip(sp_label.species, sp_label.sdm_label):
    top100_sp[top100 == label] = species
    truth_sp[labels == label] = species

# get rank of prediction
sorted_preds = np.argsort(-predictions, axis=1)
rank_preds = []
for index, label in enumerate(labels):
    rank_preds.append(np.where(sorted_preds[index, :] == label)[0][0])
print(rank_preds)



# evaluate
print('Final test:')
print(evaluate(predictions, labels, METRICS, final=True))
