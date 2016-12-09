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
    target_sentence_vector_length = 30

    def __init__(self):
        self.pca = PCA(n_components=ISentenceEmbedding.target_sentence_vector_length)

    def build(self, word_embedding, labels, sentences):
        """
        A wrapper for build_raw which performs further preprocessing on embedding
        """
        self.build_raw(word_embedding, labels, sentences)
        self.pca.fit([self.get_normalized(sentence) for sentence in sentences])

    @abstractmethod
    def build_raw(self, word_embedding, labels, sentences):
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

    def get_normalized(self, sentence):
        vector = self.get_raw_vector(sentence)

        # normalize result vector
        result_norm = (sum(map(lambda x: x**2, vector))) ** 0.5
        if result_norm != 0:
            for i in xrange(vector.shape[0]):
                vector[i] /= result_norm
        return vector

    def __getitem__(self, sentence):
        """
        A wrapper for get_raw_vector which returns vector after preprocessing
        """
        return self.pca.transform([self.get_normalized(sentence)])[0]

    @abstractmethod
    def get_raw_vector(self, sentence):
        """
        Returns vector representation for a given sentence based on current model
        :param sentence: sentence to be vectorized
        :type sentence: list of strings (words)
        :return: vector representation of the sentence, formatted as numpy vector of doubles
        """
        raise NotImplementedError
