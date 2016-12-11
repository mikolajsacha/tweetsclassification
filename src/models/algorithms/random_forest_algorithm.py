"""
Contains class for usage of Random Forest method
"""
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn.ensemble import RandomForestClassifier


class RandomForestAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Random Forest method
    """

    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding
        self.clf = RandomForestClassifier(**kwargs)

    def fit(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]

