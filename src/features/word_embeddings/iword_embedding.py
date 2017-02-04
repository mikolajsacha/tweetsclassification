"""
Contains basic interface (abstract base class) for word embeddings.
"""
import os
from abc import ABCMeta, abstractmethod

class IWordEmbedding(object):
    """
    Abstract base class for word embeddings
    """
    __metaclass__ = ABCMeta

    def __init__(self, path, vector_length):
        self.model = None
        self.path = path
        self.vector_length = vector_length
        self.already_built = False

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, word):
        raise NotImplementedError

    def build(self):
        """ Loads word embedding from its file """
        if not self.already_built:
            print("Loading pre-trained word embedding from {0}...".format(self.path))
            self._build()
            self.already_built = True
            print("Pre-trained word embedding from {0} loaded!".format(self.path))


    def get_embedding_model_path(self):
        """ :return: absolute path to folder containing saved word embedding model """
        return os.path.join(os.path.dirname(__file__), '../../../models/word_embeddings', self.path)

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
