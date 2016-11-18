"""
Contains classes for creating sentece embeddings based on given word embeddings
"""
from src.features.sentence_embeddings.isentence_embedding import ISentenceEmbedding
from src.features.word_embeddings.iword_embedding import IWordEmbedding
import numpy as np


class ConcatenationEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for senteces by simply concatenating word vectors from a given word embedding
    """
    def __init__(self):
        self.word_embedding = None
        self.vector_length = 0
        self.sentences_length = 0
        self.max_sentence_length = 0

    def build(self, word_embedding, sentences):
        self.word_embedding = word_embedding
        self.max_sentence_length = reduce(lambda acc, x: max(acc, len(x)), sentences, 0)
        self.vector_length = IWordEmbedding.target_vector_length * self.max_sentence_length

    def __getitem__(self, sentence):
        empty_vectors = self.max_sentence_length - len(sentence)
        return np.concatenate(map(lambda word: self.word_embedding[word], np.append(sentence, ([''] * empty_vectors))))

    def get_vector_length(self):
        return self.vector_length


class AverageEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by adding word vectors coordinates and averaging them
    """

    def __init__(self):
        self.word_embedding = None

    def build(self, word_embedding, sentences):
        self.word_embedding = word_embedding

    def __getitem__(self, sentence):
        vector_size = self.word_embedding.target_vector_length
        words_count = float(len(sentence))
        word_vectors = map(lambda word: self.word_embedding[word], sentence)
        result = np.empty(vector_size, dtype=float)
        for i in xrange(vector_size):
            result[i] = sum(map(lambda w: w[i], word_vectors)) / words_count
        return result

    def get_vector_length(self):
        return IWordEmbedding.target_vector_length
