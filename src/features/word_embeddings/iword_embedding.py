"""
Contains basic interface (abstract base class) for word embeddings.
"""
from abc import ABCMeta, abstractmethod
from src.data.make_dataset import get_processed_data_path
import os
import numpy as np


class IWordEmbedding(object):
    """
    Abstract base class for word embeddings
    """
    __metaclass__ = ABCMeta
    initial_vector_length = 100  # vector length through embedding process
    target_vector_length = 10  # vector length after preprocessing of embedding

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
    def saved_embedding_exists(self, data_folder):
        """
        :param data_folder: name of folder of data set (e. g. 'dataset1')
        :type data_folder: string (folder name)
        :return: True/False indicating if there exists saved file with this embedding
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, data_folder):
        """
        Saves current model to a file located in proper direcotry
        :param data_folder: name of folder of data set (e. g. 'dataset1')
        :type data_folder: string (folder name)
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, data_folder, sentences):
        """
        Loads model from a file locaten in a proper directory for data_folder name
        :param data_folder: name of folder of data set (e. g. 'dataset1')
        :param sentences: list of training set sentences (to build preprocessing transformation)
        :type data_folder: string (folder name)
        :type sentences: list of lists of strings (words)
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

    @staticmethod
    def get_embedding_model_path(data_folder):
        """
        :param data_folder: name of folder of data set (e. g. 'dataset1')
        :type data_folder: string (folder name)
        :return: absolute path to folder containing saved word embedding model
        """
        return os.path.join(os.path.dirname(__file__), '..\\..\\..\\models\\word_embeddings\\' + data_folder)

    @staticmethod
    def data_file_to_sentences(data_file_path):
        """
        Converts a processed data file to generator of lists of words
        :param data_file_path: path to data file
        :return: iterator yielding sentences as lists of words
        """
        with open(data_file_path, 'r') as f:
            for line in f:
                sentence = line.split(' ')[1]
                yield map(lambda word: word.rstrip(), sentence.split(','))

    def build_from_data_set(self, data_folder, vector_length):
        """
        Loads model from a processed data set in given data folder
        :param data_folder: name of folder of data set (e. g. 'dataset1')
        :param vector_length: length of vector in word embedding
        :type data_folder: string (folder name)
        :type vector_length: non-negative integer
        """
        data_file_path = get_processed_data_path(data_folder)
        sentences = list(self.data_file_to_sentences(data_file_path))
        self.build(sentences, vector_length)

    def sentence_to_vector(self, sentence):
        """
        :param sentence: a sentence (list of words)
        :type sentence: list of strings (words)
        :return: numpy vector build as concatenation of vectors representing sentence
        """
        return np.concatenate(map(lambda word: self[word], sentence))

