"""
Contains basic interface (abstract base class) for sentence embeddings.
"""
from abc import ABCMeta, abstractmethod


class ISentenceEmbedding(object):
    """
    Abstract base class for sentece embeddings.
    Sentence embedding creates vectors representing sentences (word lists) using a specified word embedding.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, word_embedding, sentences):
        """
        Generates sentence embedding for a given word embedding
        :param sentences: a vector of sentences
        :param word_embedding: word embedding, for which sentence embedding will be built
        :type word_embedding: an instance of class implementing IWordEmbedding interface
        :type sentences: a numpy object vector (vector of lists)
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, sentence):
        """
        Returns vector representation for a given sentence based on current model
        :param sentence: sentence to be vectorized
        :type sentence: list of strings (words)
        :return: vector representation of the sentence, formatted as numpy vector of doubles
        """
        raise NotImplementedError

    @abstractmethod
    def get_vector_length(self):
        raise NotImplementedError
