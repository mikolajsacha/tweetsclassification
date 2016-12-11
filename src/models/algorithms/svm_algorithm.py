"""
Contains class for usage of SVM algorithm
"""
from sklearn import svm
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """
    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding
        self.clf = svm.SVC(**kwargs)

    def fit(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]
