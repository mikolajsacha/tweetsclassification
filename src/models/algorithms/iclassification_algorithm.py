"""
Contains basic interface (abstract base class) for classification algorithms
"""
from abc import ABCMeta, abstractmethod
import json


class IClassificationAlgorithm(object):
    """
    Abstract base class for classification algorithms
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, labels, features, embedding, data_set_info_path):
        """
        Generates model for predicting categories using specific method
        :param labels: list of labels (categories) of sentences
        :param features: matrix of features as real values
        :param embedding: embedding used in model
        :param data_set_info_path: path to JSON file containing information about data set
        :type labels: list of non-negative integers
        :type features: numpy matrix of floats
        :param embedding: instance of class deriving IWordEmbedding
        :param data_set_info_path: string (path to JSON file)
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

    @staticmethod
    def read_data_info(data_set_info_path):
        with open(data_set_info_path) as data_file:
            return json.load(data_file)

