"""
Contains class for usage of Random Forest method
"""
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn import neighbors
import numpy as np


class NearestNeighborsAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using K - Nearest Neighbors method
    """

    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding

        args = NearestNeighborsAlgorithm.kwargs_to_args(kwargs)
        # suppress this param, as neural network doesn't use such parameter
        if 'probability' in kwargs:
            del kwargs['probability']
        self.clf = neighbors.KNeighborsClassifier(*args, **kwargs)

    @staticmethod
    def kwargs_to_args(kwargs):
        default_arguments = [("n_neighbors", 5), ("weights", 'uniform'), ("algorithm", 'auto'), ("leaf_size", 30),
                             ("p", 2), ("metric", 'minkowski'), ("metric_params", None)]
        result_arguments = []
        for arg, default_val in default_arguments:
            if arg in kwargs:
                result_arguments.append(kwargs[arg])
                del kwargs[arg]
            else:
                result_arguments.append(default_val)
        return result_arguments

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

