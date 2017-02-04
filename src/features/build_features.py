"""
Contains class FeatureBuilder for building feature set from given data set and word embedding
"""
import numpy as np


class FeatureBuilder(object):
    """
    Class used for building feature matrix.
    Field "labels" is a list of categories of sentences
    Field "features" is a features matrix of shape (training set sixe, vector_length)
    """

    def __init__(self):
        self.labels = np.empty(0, dtype=np.uint8)
        self.features = np.empty(0, dtype=float)
        self.labels.flags.writeable = False
        self.features.flags.writeable = False

    def build(self, sentence_embedding, labels, sentences):
        """
        :param sentence_embedding: instance of sentence embedding class implementing ISentenceEmbedding interface
        :param labels: a numpy vector of labels of sentences
        :param sentences: a numpy matrix of sentences (rows = sentences, columns = words)
        """
        self.labels = labels
        sentences_vectors_length = sentence_embedding.target_vector_length
        self.features = np.empty((sentences.shape[0], sentences_vectors_length), dtype=float)

        for i in xrange(sentences.shape[0]):
            self.features[i] = sentence_embedding[sentences[i]]

        self.labels.flags.writeable = False
        self.features.flags.writeable = False

