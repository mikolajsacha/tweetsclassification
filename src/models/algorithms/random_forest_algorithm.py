"""
Contains class for usage of Random Forest method
"""
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomForestAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Random Forest method
    """

    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding

        # suppress this param, as random forest doesn't use such parameter
        if 'probability' in kwargs:
            del kwargs['probability']

        self.clf = RandomForestClassifier(**kwargs)

    def fit(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]

    def visualize_2d(self, xs, ys, ax, color_map):
        estimator_alpha = 1.0 / len(self.clf.estimators_)

        for tree in self.clf.estimators_:
            z = tree.predict(np.c_[xs.ravel(), ys.ravel()])
            z = z.reshape(xs.shape)
            ax.contourf(xs, ys, z, alpha=estimator_alpha, cmap=color_map)

