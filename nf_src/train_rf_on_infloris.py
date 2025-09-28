# This script is OBSOLETE, use ./nf_src/statistical_modeling_main.py instead

# caution: this script was initially written for Windows, paths may need to be adapted for other OS
# caution: this script was initially located on the parent folder /../, paths may need to be adapted if moved

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
import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Training on Infloris dataset (F. Gourgues data only), testing on a subsample (non-F. Gourgues)

# SETTINGS
# files
# TODO: transfer these paths to the config file
TRAINSET_PATH = Path(r'C:\Users\fauren\PycharmProjects\bac_a_sable\GP\databases\InFloris\infloris_reformat.csv')
TESTSET_PATH = Path(r'C:\Users\fauren\PycharmProjects\bac_a_sable\GP\databases\InFloris\infloris_reformat.csv')
RASTER_PATH = Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm\data\rasters_GLC19')   # './data/rasters_GLC19/'

#TODO: transfer these settings to the config file
n_test = 40000
do_large = False
do_train = False

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
MAX_DEPTH = 14
N_TREES = 100

# evaluation
METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30, 100, 452]), ValidationAccuracyMultiple([1, 10, 30, 100, 452]))


# READ DATASET
df = pd.read_csv(TRAINSET_PATH, header='infer', sep=';', low_memory=False)

# clean-up dataframe
df = df.loc[df.sdm_label >= 0, :]
df = df.loc[df.Note >= 0, :]
df['id'] = df.index

# select a subset
df = df.loc[df.Observateur == 'Frédéric GOURGUES', :]



ids = df['id'].to_numpy()
labels = df['sdm_label'].astype('uint16').to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()

# extracting frequencies

freqs = np.unique(labels, return_counts=True)

sorted_labels = freqs[0][np.argsort(-freqs[1])] # labels sorted by decreasing order of frequencies


# splitting train val test
train_labels, val_labels, train_positions, val_positions, train_ids, val_ids\
    = train_test_split(labels, positions, ids, test_size=VAL_SIZE, random_state=42)

# create patch extractor
extractor = PatchExtractor(str(RASTER_PATH), size=PATCH_SIZE, verbose=True)
# add all default rasters
tic = time.time()
extractor.add_all()
print(f'Time to add Raster layers: {time.time() - tic:.3f} seconds')

if do_train:
    # constructing pytorch dataset
    train_set = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor)
    validation_set = EnvironmentalDataset(val_labels, val_positions, val_ids, patch_extractor=extractor)

    # CONSTRUCT MODEL
    clf = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, n_jobs=16)

    # TRAINING

    X, y = train_set.numpy()
    print(f'Training on {y.shape[0]} Examples')

    tic = time.time()
    clf.fit(X, y)
    print(f'Time to fit model: {time.time() - tic:.3f} seconds')

    # save model
    path = 'rf_infloris.skl'
    print('Saving SKL model: ' + path)
    joblib.dump(clf, path)


    # VALIDATION
    print('\nValidation: ')
    inputs, labels = validation_set.numpy()
    restricted_predictions = clf.predict_proba(inputs)
    # sklearn fit() doesn't take as input the labels or the number of classes. It's infer with the training data.
    # Labels order in prediction of the clf model is given in clf.classes_.
    # With random split, some species can be only on the test or validation set and will not be present in the prediction.
    # So, the prediction need to be reshape to covers all species as following.
    predictions = np.zeros((restricted_predictions.shape[0], N_LABELS))
    predictions[:, clf.classes_] = restricted_predictions

    print(evaluate(predictions, labels, METRICS))


# FINAL EVALUATION ON TEST SET
# read test set
df = pd.read_csv(TESTSET_PATH, header='infer', sep=';', low_memory=False)

# clean-up dataframe
df = df.loc[df.sdm_label >= 0, :]
# df = df.loc[df.Note >= 0, :] # suppress non-noted observations
df['id'] = df.index

df = df.loc[df.Observateur != 'Frédéric GOURGUES', :]
df = df.iloc[np.random.choice(np.arange(df.shape[0]), n_test, replace=False), :] # temporary restriction to a random subset

ids = df['id'].to_numpy()
labels = df['sdm_label'].astype('uint16').to_numpy()
positions = df[[LATITUDE, LONGITUDE]].to_numpy()
test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)

# load model
def predict_from_model(path='rf_infloris.skl', dataset=test_set, n_labels=N_LABELS):
    print('Loading SKL model: ' + path)
    clf = joblib.load(path)

    # predict
    inputs, labels = dataset.numpy()
    restricted_predictions = clf.predict_proba(inputs)
    predictions = np.zeros((restricted_predictions.shape[0], n_labels))
    predictions[:, clf.classes_] = restricted_predictions

    return predictions

# evaluate
print('\nFinal test:')
predictions = predict_from_model()
print(evaluate(predictions, labels, METRICS, final=True))

# Format and Save predictions
# convert predictions to top 100
path2sp_label = Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm\data\species_label_table.csv')
sp_label = pd.read_csv(path2sp_label, sep=';')

sorted_preds = np.argsort(-predictions, axis=1)
top100 = sorted_preds[:, :100]
top100_sp = np.empty(top100.shape, dtype=object)
truth_sp = np.empty(labels.shape, dtype=object)

for species, label in zip(sp_label.species, sp_label.sdm_label):
    top100_sp[top100 == label] = species
    truth_sp[labels == label] = species

# get rank of prediction
sorted_preds = np.argsort(-predictions, axis=1)
rank_preds = []
rank_freqs = []
for index, label in enumerate(labels):
    rank_preds.append(1 + np.where(sorted_preds[index, :] == label)[0][0])
    try:
        rank_freqs.append(1 + np.where(sorted_labels[:] == label)[0][0])
    except IndexError: # dealing with case where species not present in train set
        rank_freqs.append(1 + sorted_labels.shape[0] + int(np.random.random(1)[0] * (4520 - sorted_labels.shape[0])))
df['rank_preds'] = rank_preds
df['rank_freqs'] = rank_freqs

# Compare with previous model

predictions_sdmdataset = predict_from_model(path='rf.skl')
sorted_preds_sdmdataset = np.argsort(-predictions_sdmdataset, axis=1)
rank_preds_sdmdataset = []
for index, label in enumerate(labels):
    rank_preds_sdmdataset.append(1 + np.where(sorted_preds_sdmdataset[index, :] == label)[0][0])
df['rank_preds_sdmdataset_model'] = rank_preds_sdmdataset

df.to_csv(Path(rf'C:\Users\fauren\PycharmProjects\cnn-sdm\output\preds_rf_infloris_14-test_set_{n_test}_small.csv'), sep=';', index=False)

if do_large:
    for index in range(100):
        df[f'{index + 1}/100'] = top100_sp[:, index]

    for species, label in zip(sp_label.species, sp_label.sdm_label):
        df[f'{species}'] = predictions[:, label]

    df.to_csv(Path(rf'C:\Users\fauren\PycharmProjects\cnn-sdm\output\preds_rf_infloris_14-test_set_{n_test}.csv'), sep=';', index=False)


print('Done !')
