"""
Contains classes for creating sentece embeddings based on given word embeddings
"""

from src.features.sentence_embeddings.isentence_embedding import ISentenceEmbedding
import numpy as np


class ConcatenationEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for senteces by simply concatenating word vectors from a given word embedding
    """

    def __init__(self):
        self.word_embedding = None

    def build(self, word_embedding):
        self.word_embedding = word_embedding

    def __getitem__(self, sentence):
        return np.concatenate(map(lambda word: self.word_embedding[word], sentence))


class AverageEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by adding word vectors coordinates and averaging them
    """

    def __init__(self):
        self.word_embedding = None

    def build(self, word_embedding):
        self.word_embedding = word_embedding

    def __getitem__(self, sentence):
        vector_size = self.word_embedding.target_vector_length
        words_count = float(len(sentence))
        word_vectors = map(lambda word: self.word_embedding[word], sentence)
        result = np.empty(vector_size, dtype=float)
        for i in xrange(vector_size):
            result[i] = sum(map(lambda w: w[i], word_vectors)) / words_count
        return result
