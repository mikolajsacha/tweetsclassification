"""
Contains classes for creating sentence embeddings based on given word embeddings
"""
import numpy as np
from src.features.sentence_embeddings.isentence_embedding import ISentenceEmbedding


class ConcatenationEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by simply concatenating word vectors from a given word embedding
    """

    def __init__(self, target_sentence_vector_length=None, max_sentence_length=30):
        ISentenceEmbedding.__init__(self, target_sentence_vector_length)
        self.max_sentence_length = max_sentence_length
        self.word_embedding = None
        self.word_vector_length = None

    def build_raw(self, word_embedding):
        self.word_embedding = word_embedding
        self.word_vector_length = word_embedding.vector_length * self.max_sentence_length
        if not self.use_pca:
            self.target_vector_length = self.word_vector_length

    def get_raw_vector(self, sentence):
        result = np.zeros((self.word_vector_length), dtype=float)
        i = 0
        for word in sentence:
            embedding = self.word_embedding[word]
            if embedding is not None:
                result[i:i+len(embedding)] = embedding
                i += len(embedding)
        return result


class SumEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by adding word vectors coordinates
    """

    def __init__(self, target_sentence_vector_length=None):
        ISentenceEmbedding.__init__(self, target_sentence_vector_length)
        self.word_embedding = None
        self.word_vector_length = None

    def build_raw(self, word_embedding):
        self.word_vector_length = word_embedding.vector_length
        self.word_embedding = word_embedding
        if not self.use_pca:
            self.target_vector_length = self.word_vector_length

    def get_raw_vector(self, sentence):
        result = np.zeros((self.word_vector_length), dtype=float)
        for word in sentence:
            embedding = self.word_embedding[word]
            if embedding is not None:
                for i, val in enumerate(embedding):
                    result[i] += val
        return result
