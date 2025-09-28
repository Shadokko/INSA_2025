__HELP__ = """
Main script to train and/or evaluate a sdm model. Adapted from 'cnn-sdm/train_rf_on_infloris.py'

usage:
first of all, copy the template config file:
cp ./input/parameters_template.yml ./input/parameters.yml ./output/experiments/<my_experiment_subfolder>



python ./nf_src/statistical_modeling_main.py -c ./input/parameters_template.yml

todo: allow for selection of rasters
"""

import sys

# import modules
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv


from lib.dataset import EnvironmentalDataset
from lib.raster import PatchExtractor
from lib.evaluation import evaluate
from lib.metrics import ValidationAccuracyMultipleBySpecies
from lib.metrics import ValidationAccuracyMultiple
from lib.freq_model.freq_model import FrequencyModel, RandomModel

import torch
from lib.cnn.utils import load_model_state
from lib.cnn.models.inception_env import InceptionEnv
from lib.cnn.predict import predict
from lib.cnn.train import fit


from pathlib2 import Path
import time
import argparse
from ruamel.yaml import YAML
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def cli_manager():
    """
    CLI Manager parse the command line interface of th script, with short and long options w/o arguments.
    It relies on argparse, the built-in script argument parser of Python
    :return: parsing results from argv by argparse
    """

    # Display a thorough description of the script
    _parser = argparse.ArgumentParser(description=__HELP__)

    # Get the parameter file using the option -c|--config
    _parser.add_argument("-c", "--config", dest="config_file", type=str, required=True, metavar="config/milcnn.yml",
                         help="Path to a config file in YAML format. Check the config/ subdirectory for examples")

    return _parser.parse_args()

def filter_dataset(df, param_dict):
    # filter the dataset
    for column, include, exclude in zip(param_dict['columns'],
                                        param_dict['inclusion'],
                                        param_dict['exclusion']):
        if column in df.columns:
            if include: df = df.loc[df[column].isin(include), :]
            if exclude: df = df.loc[~df[column].isin(exclude), :]
        else:
            raise Warning(f"Column {column} is not present in the dataset"
                          f"\n    -> it will not be used for filtering")

    if param_dict['random_subset']:
        np.random.seed(param_dict['seed'])
        df = df.iloc[np.random.choice(np.arange(df.shape[0]), param_dict['random_subset'], replace=False), :]
    return df

def get_id_label_pos(df, param_dict):
    ids = df[param_dict['id_column']].to_numpy()
    labels = df[param_dict['label_column']].astype('uint16').to_numpy()
    positions = df[[param_dict['latitude_column'], param_dict['longitude_column']]].to_numpy()

    return ids, labels, positions

def predict_from_model(clf, dataset, n_labels):

    # predict
    inputs, labels = dataset.numpy()
    restricted_predictions = clf.predict_proba(inputs)
    predictions = np.zeros((restricted_predictions.shape[0], n_labels))
    predictions[:, clf.classes_] = restricted_predictions

    return predictions

def export_predictions(df, predictions, labels, path2outputfile, path2sp_label, label_column='Rang_frequence_espece', sp_column='Espece',
                       do_top100=False, do_raw_predictions=False, verbose=True):
    """
    Generates a summary of results and concatenates it to the raw metadata.

    Parameters
    ----------
    df: pandas DataFrame, Ground truth with all metadata
    predictions: numpy array, a table of probabilities corresponding to predictions by instance (rows) and label (columns)
    labels: numpy array of labels corresponding to each row. Is somewhat redundant, used as control.
    path2outputfile: path to the file where the output will be saved
    path2sp_label: path to a file with label/species correspondance.
    do_top100: boolean, whether to generate/export 100 columns with species names corresponding to top 100 predictions
    do_raw_predictions: boolean, whether to generat export rax predictions, sorted by labels (number of columns added equal to number of labels)
    verbose: boolean
    """
    # add labels to dataframe (for control purposes, they should be already present)
    df['labels_used_for_test'] = labels

    # get rank and predicted probability of ground truth species
    sorted_preds = np.argsort(-predictions, axis=1)  # prediction labels sorted by rank
    rank_preds = []
    proba_preds = []
    for index, label in enumerate(labels):
        rank_preds.append(1 + np.where(sorted_preds[index, :] == label)[0][0])
        proba_preds.append(predictions[index, label])
    df['rank_ground_truth'] = rank_preds
    df['proba_ground_truth'] = proba_preds

    if do_top100:
        # convert predictions to top 100
        sp_label = pd.read_csv(path2sp_label, sep=';')

        top100 = sorted_preds[:, :100]
        top100_sp = np.empty(top100.shape, dtype=object)
        truth_sp = np.empty(labels.shape, dtype=object)

        for species, label in zip(sp_label[sp_column], sp_label[label_column]):
            top100_sp[top100 == label] = species
            truth_sp[labels == label] = species

        for index in range(100):
            df[f'{index + 1}/100'] = top100_sp[:, index]

    if do_raw_predictions:
        for species, label in zip(sp_label[sp_column], sp_label[label_column]):
            try:
                df[f'{species}'] = predictions[:, label]
            except IndexError: # dealing with the case where more labels are provided than used (Case of Infloris top1200)
                pass

    # saving
    if verbose:
        print(f'Exporting results to {Path(path2outputfile)}')
    df.to_csv(Path(path2outputfile), sep=';', index=False)

def construct_inception_model(extractor, DROPOUT=0.7, N_LABELS=4520):
    # CONSTRUCT MODEL
    patch = extractor[(45.17795, 5.7146)]  # generating a patch, just to check and extract input shape
    n_input = patch.shape[0]

    model = InceptionEnv(dropout=DROPOUT, n_labels=N_LABELS, n_input=n_input)

    if torch.cuda.is_available():
        # check if GPU is available
        print("Training on GPU")
        device = torch.device('cuda')
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    else:
        print("Training on CPU")

    return model


def load_inception_model(path2savedmodel, DROPOUT=0.7, N_LABELS=4520):

    model = InceptionEnv(dropout=DROPOUT, n_labels=N_LABELS)
    load_model_state(model, str(path))
    if torch.cuda.is_available():
        # check if GPU is available
        model.to(torch.cuda.device(0))
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])

    return model

def predict_inception(clf, test_set):
    return predict(clf, test_set)

###########################################

if __name__ == '__main__':
    # parsing arguments and loading parameters
    cli = cli_manager()

    # Some verbosity and preparation
    print(f'Executing script "{sys.argv[0]}" \n    with argument {sys.argv[1]}')
    print(f'Time of launch: {time.asctime()}')
    tics =  {}
    tics['start'] = time.time()
    yaml = YAML(typ='safe')
    param = yaml.load(Path(cli.config_file))

    os.chdir(param['global']['root_path'])
    print(f'Working directory is now: {os.getcwd()}')

    # set metrics (todo: make it parametrizable)
    METRICS = (ValidationAccuracyMultipleBySpecies([1, 10, 30, 100, 452]), ValidationAccuracyMultiple([1, 10, 30, 100, 452]))

    tics['patch_extractor'] = time.time()
    # create patch extractor
    extractor = PatchExtractor(str(param['global']['raster_path']),
                               size=param['global']['patch_size'], verbose=True)
    # add rasters
    tic = time.time()
    try:
        add_T = param['global']['add_T']
        extractor.add_all(add_T=add_T) # raster_metadata=param['global']['raster_metadata'])
    except KeyError:
        extractor.add_all()
    print(f"Time to add Raster layers: {time.time() - tics['patch_extractor']:.2f} seconds")

    # training
    if param['train']['do_train']:
        print('\nDataset building...')
        tics['dataset_building'] = time.time()
        # read training dataset
        df = pd.read_csv(Path(param['train']['trainset_path']), header='infer', sep=';', low_memory=False)

        # filter the dataset
        df = filter_dataset(df, param['train']['sampling'])

        ids, labels, positions = get_id_label_pos(df, param['train'])

        # constructing dataset
        # splitting train val test
        train_labels, val_labels, train_positions, val_positions, train_ids, val_ids \
            = train_test_split(labels, positions, ids, test_size=param['train']['sampling']['val_size'], random_state=42)
        train_set = EnvironmentalDataset(train_labels, train_positions, train_ids, patch_extractor=extractor)
        validation_set = EnvironmentalDataset(val_labels, val_positions, val_ids, patch_extractor=extractor)

        X, y = train_set.numpy()
        print(f"Time to build dataset: {time.time() - tics['dataset_building']:.2f} seconds")
        print(f'Training on {y.shape[0]} Examples')

        # CONSTRUCT MODEL
        if param['train']['model']['type'] == 'rf':
            clf = RandomForestClassifier(n_jobs=16, **param['train']['model']['parameters'])
        elif param['train']['model']['type'] == 'frequency':
            clf = FrequencyModel()
        elif param['train']['model']['type'] == 'random':
            clf = RandomModel(n_classes=param['global']['n_label'])
        elif param['train']['model']['type'] == 'inception':
            clf = construct_inception_model(extractor, N_LABELS=param['global']['n_label'])

        # training model
        print('\nTraining...')
        tics['training'] = time.time()
        if param['train']['model']['type'] == 'inception':
            fit(clf,
                train=train_set, validation=validation_set,
                metrics=(ValidationAccuracyMultipleBySpecies([1, 10, 30, 100]), ValidationAccuracyMultiple([1, 10, 30, 100])),
                **param['train']['model']['parameters'])
        elif param['train']['model']['type'] == 'blbla': # set to an unlikely name if you want to disable
            print("Under dev: cross-validation")
            parameters = {'n_estimators': [300],
                          'criterion': ['gini'],
                          'max_features': [7, 8, 9],
                         'max_depth': [8, 9, 10, 11]
                          }
            scorer = sklearn.metrics.make_scorer(sklearn.metrics.top_k_accuracy_score, needs_proba=True, k=30)
            clf = sklearn.model_selection.HalvingGridSearchCV(clf, param_grid=parameters, scoring=scorer,
                                                               min_resources=100000, factor=6, cv=3, verbose=3, random_state=38)
            clf.fit(X, y)
            print(f'Best parameters found: ')
            print(clf.best_estimator_)

        else:
            clf.fit(X, y)
        print(f"Time to fit model: {time.time() - tics['training']:.2f} seconds")

        # saving model
        path = param['train']['model']['save']
        if path:
            print(f'Saving SKL model: {path}')
            if param['train']['model']['type'] == 'rf':
                joblib.dump(clf, path)
            elif param['train']['model']['type'] == 'frequency':
                clf.save(path)
            elif param['train']['model']['type'] == 'random':
                clf.save(path)
            elif param['train']['model']['type'] == 'inception':
                torch.save(clf, path)

        # VALIDATION
        print('\nValidation: ')
        inputs, labels = validation_set.numpy()
        restricted_predictions = clf.predict_proba(inputs)
        # sklearn fit() doesn't take as input the labels or the number of classes. It's infer with the training data.
        # Labels order in prediction of the clf model is given in clf.classes_.
        # With random split, some species can be only on the test or validation set and will not be present in the prediction.
        # So, the prediction need to be reshape to covers all species as following.
        predictions = np.zeros((restricted_predictions.shape[0],
                                param['global']['n_label']))
        predictions[:, clf.classes_] = restricted_predictions

        print(evaluate(predictions, labels, METRICS))

    # testing / evaluation
    if param['test']['do_test']:
        # some clean-up for memory economy
        train_set = None
        validation_set = None
        X = None
        df = None

        print(f'load and filter the test set...')
        df_test = pd.read_csv(param['test']['testset_path'], header='infer', sep=';', low_memory=False)
        df_test = filter_dataset(df_test, param['test']['sampling'])

        ids, labels, positions = get_id_label_pos(df_test, param['test'])
        test_set = EnvironmentalDataset(labels, positions, ids, patch_extractor=extractor)

        print('\nEvaluation/testing...')
        if param['test']['model_path']:
            path = param['test']['model_path']
        else:
            path = param['train']['model']['save'] # todo: change to avoid reloading model if not needed

        print('Loading model: ' + path)
        if param['train']['model']['type'] == 'rf':
            clf = joblib.load(path)
        elif param['train']['model']['type'] == 'frequency':
            clf = FrequencyModel(path)
        elif param['train']['model']['type'] == 'random':
            clf = RandomModel(path2savedmodel=path)
        elif param['train']['model']['type'] == 'inception':
            clf = load_inception_model(path2savedmodel=path, N_LABELS=param['global']['n_label'])

        print('Predicting')
        if param['train']['model']['type'] == 'inception':
            predictions, labels = predict_inception(clf, test_set)
        else:
            predictions = predict_from_model(clf, test_set, param['global']['n_label'])

        # evaluating
        print(evaluate(predictions, labels, METRICS, final=True))

        # Export predictions
        print('exporting predictions')
        export_predictions(df_test, predictions, labels,
                           param['test']['export']['path2outputfile'],
                           param['test']['export']['path2sp_label'],
                           label_column=param['test']['export']['label_column'],
                           sp_column=param['test']['export']['sp_column'],
                           do_top100=param['test']['export']['do_top100'],
                           do_raw_predictions=param['test']['export']['do_raw_predictions'])

    # exit nicely
    print('Done !')
    print(f"Total execution time: {time.time() - tics['start']:.2f} seconds")
    sys.exit(0)