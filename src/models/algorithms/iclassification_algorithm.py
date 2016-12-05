"""
Contains basic interface (abstract base class) for classification algorithms
"""
from abc import ABCMeta, abstractmethod


class IClassificationAlgorithm(object):
    """
    Abstract base class for classification algorithms
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, labels, features):
        """
        Generates model for predicting categories using specific method
        :param labels: list of labels (categories) of sentences
        :param features: matrix of features as real values
        :type labels: list of non-negative integers
        :type features: numpy matrix of floats
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, sentence):
        """
        Predicts category for given sentence
        :param sentence: list of words (sentence)
        :type sentence: list of strings (words)
        :return: list of pairs (category, probability)
        """
        raise NotImplementedError

    @abstractmethod
    def get_estimator(self):
        """
        :return: An estimator object from sklearn package for this algorithm
        """
        raise NotImplementedError
