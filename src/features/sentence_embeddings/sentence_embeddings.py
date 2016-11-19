"""
Contains classes for creating sentece embeddings based on given word embeddings
"""
import numpy as np
from collections import Counter
from src.features.sentence_embeddings.isentence_embedding import ISentenceEmbedding
from src.features.word_embeddings.iword_embedding import IWordEmbedding


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


class TermFrequencyAverageEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by averaging word vectors, where more frequent have lower weights
    """

    def __init__(self):
        self.word_embedding = None
        self.weights = {}
        self.min_weight = 0

    def build(self, word_embedding, sentences):
        self.word_embedding = word_embedding
        word_counter = Counter(word for sentence in sentences for word in sentence)
        total_words_count = float(reduce(lambda acc, x: acc + len(x), sentences, 0))
        for word, occurrences in word_counter.iteritems():
            self.weights[word] = (total_words_count - occurrences) / total_words_count
        self.min_weight = 1.0 / total_words_count

    def get_weight(self, word):
        if word not in self.weights:
            return self.min_weight
        return self.weights[word]

    def __getitem__(self, sentence):
        vector_size = self.word_embedding.target_vector_length
        words_count = float(len(sentence))
        word_vectors = map(lambda word: self.word_embedding[word], sentence)
        result = np.zeros(vector_size, dtype=float)
        for i in xrange(vector_size):
            for j, word in enumerate(sentence):
                result[i] += self.get_weight(word) * word_vectors[j][i]
            result[i] /= words_count
        return result

    def get_vector_length(self):
        return IWordEmbedding.target_vector_length
