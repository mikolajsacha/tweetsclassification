"""
Contains basic interface (abstract base class) for sentence embeddings.
"""
from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA


class ISentenceEmbedding(object):
    """
    Abstract base class for sentece embeddings.
    Sentence embedding creates vectors representing sentences (word lists) using a specified word embedding.
    """
    __metaclass__ = ABCMeta

    def __init__(self, target_sentence_vector_length):
        self.use_pca = target_sentence_vector_length is not None
        if self.use_pca:
            self.target_vector_length = target_sentence_vector_length
            self.pca = PCA(n_components=self.target_vector_length)

    def build(self, word_embedding, sentences = None):
        """
        A wrapper for build_raw which performs further preprocessing on embedding
        """
        self.build_raw(word_embedding)
        if self.use_pca:
            if sentences is None:
                raise AttributeError("When using PCA you must provide training sentences to sentence embedding")
            self.pca.fit([self.get_raw_vector(sentence) for sentence in sentences])

    @abstractmethod
    def build_raw(self, word_embedding):
        """
        Generates sentence embedding for a given word embedding
        :param word_embedding: word embedding, for which sentence embedding will be built
        :type word_embedding: an instance of class implementing IWordEmbedding interface
        """
        raise NotImplementedError

    def __getitem__(self, sentence):
        """
        A wrapper for get_raw_vector which returns vector after preprocessing
        """
        if self.use_pca:
            return self.pca.transform([self.get_raw_vector(sentence)])[0]
        else:
            return self.get_raw_vector(sentence)

    @abstractmethod
    def get_raw_vector(self, sentence):
        """
        Returns vector representation for a given sentence based on current model
        :param sentence: sentence to be vectorized
        :type sentence: list of strings (words)
        :return: vector representation of the sentence, formatted as numpy vector of doubles
        """
        raise NotImplementedError
