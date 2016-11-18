"""
Contains basic interface (abstract base class) for word embeddings.
"""
import os
import numpy as np
import nltk
from abc import ABCMeta, abstractmethod


class TextCorpora(object):
    """ Utility class for downloading text corpora using NLTK download manager and storing them in one place """
    corpuses = {}

    @staticmethod
    def get_corpus(key):
        if key not in TextCorpora.corpuses:
            text_corpus_downloaded = False
            while not text_corpus_downloaded:
                try:
                    TextCorpora.corpuses[key] = iter(eval("nltk.corpus.{0}.sents()".format(key)))
                    text_corpus_downloaded = True
                except LookupError:
                    print ("Please use NLTK manager to download text corpus \"{0}\"".format(key))
                    nltk.download()
                except AttributeError:
                    raise KeyError("There is no NLTK text corpus keyed \"{0}\"".format(key))
        return TextCorpora.corpuses[key]


class IWordEmbedding(object):
    """
    Abstract base class for word embeddings
    """
    __metaclass__ = ABCMeta
    initial_vector_length = 100  # vector length through embedding process
    target_vector_length = 50  # vector length after preprocessing of embedding

    @abstractmethod
    def build(self, sentences):
        """
        Generates word embedding for given list of sentences
        :param sentences: list of sentences in data set, formatted as lists of words
        :type sentences: list of list of strings
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
    def save(self, output_path):
        """
        Saves current embedding to a file located in proper direcotry
        :param output_path: absolute path to file where embedding should be saved
        :type output_path: string (file path)
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, data_path, sentences):
        """
        Loads model from a file locaten in a proper directory for data_folder name
        :param data_path: absolute path to a file with saved embedding
        :param sentences: list of training set sentences (to build preprocessing transformation)
        :type data_path: string (file path)
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