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
    def build(self, word_embedding, labels, sentences):
        """
        Generates sentence embedding for a given word embedding
        :param labels: a vector of labels of sentences
        :param sentences: a vector of sentences
        :param word_embedding: word embedding, for which sentence embedding will be built
        :type word_embedding: an instance of class implementing IWordEmbedding interface
        :type labels: a numpy uint32 vector
        :type sentences: a numpy object vector (vector of lists)
        """
        raise NotImplementedError

    def __getitem__(self, sentence):
        """
        A wrapper for get_raw_vector which returns vector after actuall preprocessing
        """
        raw_vector = self.get_raw_vector(sentence)

        # normalize result vector
        result_norm = (sum(map(lambda x: x**2, raw_vector))) ** 0.5
        if result_norm != 0:
            for i in xrange(raw_vector.shape[0]):
                raw_vector[i] /= result_norm
        return raw_vector

    @abstractmethod
    def get_raw_vector(self, sentence):
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
