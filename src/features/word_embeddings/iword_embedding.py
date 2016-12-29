"""
Contains basic interface (abstract base class) for word embeddings.
"""
import os
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
                    TextCorpora.corpuses[key] = list(eval("nltk.corpus.{0}.sents()".format(key)))
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

    def __init__(self, text_corpus, vector_length):
        self.vector_length = vector_length
        self.text_corpus = text_corpus

    @abstractmethod
    def build(self, sentences):
        """
        Generates word embedding for given list of sentences
        :param sentences: list of sentences in data set, formatted as lists of words
        :type sentences: list of list of strings
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

    def __str__(self):
        return type(self).__name__
