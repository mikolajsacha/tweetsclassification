"""
Contains basic interface (abstract base class) for word embeddings.
"""
from abc import ABCMeta, abstractmethod


class IWordEmbedding(object):
    """
    Abstract base class word word embeddings
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, sentences, vector_length):
        """
        Generates word embedding for given list of sentences
        :param sentences: list of sentences in data set, formatted as lists of words
        :param vector_length: length of vector in word embedding
        :type sentences: list of list of strings
        :type vector_length: non-negative integer
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, file_path):
        """
        Loads model from a given file
        :param file_path: path to file containing saved model
        :type file_path: string (file path)
        """
        raise NotImplementedError

    @abstractmethod
    def safe(self, file_path):
        """
        Safes current model to a file
        :param file_path: path where model is to be saved
        :type file_path: string (file path)
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, word):
        """
        Returns vector representation for given word based on current model
        :param word: word to be vectorized
        :type word: string
        :return: vector representation of word, formatted as list of doubles
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_data_path(self):
        """
        :return: path to file containing saved word embedding model
        """
        return '../../models/word_embeddings'
