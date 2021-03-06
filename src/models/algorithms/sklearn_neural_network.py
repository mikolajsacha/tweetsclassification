"""
Contains class for usage of Neural Network algorithm
"""
import warnings

from sklearn.neural_network import MLPClassifier
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
import numpy as np


class MLPAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Neural Network method
    """
    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding

        # suppress this param, as neural network doesn't use such parameter
        if 'probability' in kwargs:
            del kwargs['probability']
        if "n_jobs" in kwargs:
            del kwargs["n_jobs"]  # multi-threading not available here

        self.clf = MLPClassifier(**kwargs)

    def fit(self, features, labels):
        # ignore ConvergenceWarning from neural network
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]

    def visualize_2d(self, xs, ys, ax, color_map):
        z = self.clf.predict(np.c_[xs.ravel(), ys.ravel()])
        z = z.reshape(xs.shape)
        ax.contourf(xs, ys, z, cmap=color_map)
