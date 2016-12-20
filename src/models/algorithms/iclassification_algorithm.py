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
    def visualize_2d(self, xs, ys, ax, color_map):
        """
        Draw a 2d visualization of algorithm on given Xs and Ys.
        Noticee: Here, Xs and Ys are lists of built features (float vectors)
        Should work only if features are 2-dimensional
        """
        raise NotImplementedError

