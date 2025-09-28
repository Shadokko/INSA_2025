import numpy as np
import pandas as pd

class FrequencyModel():
    def __init__(self, path2frequency_table=False):
        """
        Initialization

        Dull model predicting according to frequencies, with no use of geographical information of any kind
        """
        if path2frequency_table:
            self.freqs = pd.read_csv(path2frequency_table, sep=';')
            self.generate_classes_freqs()

    def fit(self, X, y):
        """
        fits the model

        @param X: not used, just kept for consistency
        @param y: labels used for training
        """
        freqs_np = np.unique(y, return_counts=True)
        self.freqs = pd.DataFrame({'label': freqs_np[0], 'occurrences': freqs_np[1]})
        self.freqs['frequency'] = self.freqs.occurrences / np.sum(self.freqs.occurrences)

        # getting frequency ranks
        sorted_labels = freqs_np[0][np.argsort(-freqs_np[1])]  # labels sorted by decreasing order of frequencies
        rank_freqs = []
        for index, label in enumerate(self.freqs.label):
            try:
                rank_freqs.append(1 + np.where(sorted_labels[:] == label)[0][0])
            except IndexError: # dealing with case where species not present in train set
                rank_freqs.append(1 + sorted_labels.shape[0] + int(np.random.random(1)[0] * (4520 - sorted_labels.shape[0])))
        self.freqs['rank_frequency'] = rank_freqs

        # getting classes and frequencies in sklearn-like format
        self.generate_classes_freqs()

    def generate_classes_freqs(self):
        # generate classes list and related frequencies
        self.classes_ = np.sort(self.freqs.label)
        self.classes_freqs = np.array([self.freqs.frequency.loc[self.freqs.label == label].values[0]
                                       for label in self.classes_])

    def predict_proba(self, X):
        self.predictions = np.zeros((X.shape[0],
                                     self.classes_.shape[0]))

        for index in range(X.shape[0]):
            self.predictions[index, :] = self.classes_freqs

        return self.predictions

    def save(self, path):
        self.freqs.to_csv(path, sep=';', index=False)



class RandomModel():
    def __init__(self, n_classes=4520, path2savedmodel=False):
        """
        Initialization

        Super-dull model predicting according to frequencies, with no use of geographical information of any kind
        """
        if path2savedmodel:
            df = pd.read_csv(path2savedmodel, sep=';')
            self.n_classes = df.iloc[0, 0]
        else:
            self.n_classes = n_classes
        self.classes_ = np.arange(self.n_classes)

    def fit(self, X, y):
        """
        fits the model: no fitting at all in fact

        @param X: not used, just kept for consistency
        @param y: labels used for training
        """
        pass


    def predict_proba(self, X):
        self.predictions = np.zeros((X.shape[0],
                                     self.n_classes))

        for index in range(X.shape[0]):
            self.predictions[index, :] = np.random.random(size=self.n_classes)
            self.predictions[index, :] = self.predictions[index, :] / np.sum(self.predictions[index, :])

        return self.predictions

    def save(self, path):
        pd.DataFrame({'n_classes': [self.n_classes]}).to_csv(path, sep=';', index=False)






