"""
Contains class for usage of Neural Network algorithm
"""
from sklearn.neural_network import MLPClassifier
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
import numpy as np


class NeuralNetworkAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Neural Network method
    """
    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding
        self.clf = MLPClassifier(**kwargs)

    def fit(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]

    def visualize_2d(self, xs, ys, ax, color_map):
        z = self.clf.predict(np.c_[xs.ravel(), ys.ravel()])
        z = z.reshape(xs.shape)
        ax.contourf(xs, ys, z, cmap=color_map)
