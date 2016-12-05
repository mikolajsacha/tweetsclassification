"""
Contains classes for creating sentece embeddings based on given word embeddings
"""
import numpy as np
import operator
from abc import ABCMeta, abstractmethod
from collections import Counter
from src.features.sentence_embeddings.isentence_embedding import ISentenceEmbedding
from src.features.word_embeddings.iword_embedding import IWordEmbedding


#  This embedding does not work well - I won't use it in further research
class ConcatenationEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for senteces by simply concatenating word vectors from a given word embedding
    """

    def __init__(self):
        ISentenceEmbedding.__init__(self)
        self.word_embedding = None
        self.vector_length = 0
        self.sentences_length = 0
        self.max_sentence_length = 0

    def build_raw(self, word_embedding, labels, sentences):
        self.word_embedding = word_embedding
        self.max_sentence_length = reduce(lambda acc, x: max(acc, len(x)), sentences, 0)
        self.vector_length = IWordEmbedding.target_vector_length * self.max_sentence_length

    def get_raw_vector(self, sentence):
        empty_vectors = self.max_sentence_length - len(sentence)
        if empty_vectors >= 0:
            return np.concatenate(map(lambda word: self.word_embedding[word],
                                      np.append(sentence, ([''] * empty_vectors))))
        else:
            return np.concatenate(map(lambda word: self.word_embedding[word], sentence[:empty_vectors]))


class SumEmbedding(ISentenceEmbedding):
    """
    Creates vector representation for sentences by adding word vectors coordinates
    """

    def __init__(self):
        ISentenceEmbedding.__init__(self)
        self.word_embedding = None

    def build_raw(self, word_embedding, labels, sentences):
        self.word_embedding = word_embedding

    def get_raw_vector(self, sentence):
        vector_size = self.word_embedding.target_vector_length
        word_vectors = map(lambda word: self.word_embedding[word], sentence)
        result = np.empty(vector_size, dtype=float)
        for i in xrange(vector_size):
            result[i] = sum(w[i] for w in word_vectors)
        return result


class IWeightedWordEmbedding(ISentenceEmbedding):
    """
    Abstract base class for embeddings based on weighting word vectors in some way
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        ISentenceEmbedding.__init__(self)
        self.word_embedding = None
        self.weights = {}
        self.min_weight = 1.0

    @abstractmethod
    def build_raw(self, word_embedding, labels, sentences):
        raise NotImplementedError

    def get_weight(self, word):
        if word not in self.weights:
            return self.min_weight
        return self.weights[word]

    def get_raw_vector(self, sentence):
        vector_size = self.word_embedding.target_vector_length
        result = np.zeros(vector_size, dtype=float)
        for word, vector in ((w, self.word_embedding[w]) for w in sentence):
            weight = self.get_weight(word)
            for i in xrange(vector_size):
                result[i] += weight * vector[i]
        return result


class TermFrequencyAverageEmbedding(IWeightedWordEmbedding):
    """
    Creates vector representation for sentences by averaging word vectors, where more frequent have higher weights
    """

    def __init__(self):
        IWeightedWordEmbedding.__init__(self)

    def build_raw(self, word_embedding, labels, sentences):
        self.word_embedding = word_embedding
        self.weights = {}
        word_counter = Counter(word for sentence in sentences for word in sentence)
        for word, occurrences in word_counter.iteritems():
            self.weights[word] = occurrences


# This embedding does not work well - I won't use it in further research
class ReverseTermFrequencyAverageEmbedding(IWeightedWordEmbedding):
    """
    Creates vector representation for sentences by averaging word vectors, where more frequent have lower weights
    """

    def __init__(self):
        IWeightedWordEmbedding.__init__(self)

    def build_raw(self, word_embedding, labels, sentences):
        self.word_embedding = word_embedding
        self.weights = {}
        word_counter = Counter(word for sentence in sentences for word in sentence)
        total_words_count = sum(len(sen) for sen in sentences)
        for word, occurrences in word_counter.iteritems():
            self.weights[word] = (total_words_count - occurrences) / total_words_count


class TermCategoryVarianceEmbedding(IWeightedWordEmbedding):
    """
    Creates vector representation for sentences weighting words in the following way:
    If the word occurs frequently in one category, but less frequently in other, it has a high weight.
    If the word occurs with roughly same counts in all categories, it has a low weight.
    """

    def __init__(self):
        IWeightedWordEmbedding.__init__(self)
        self.sorted_words = []  # for analysis

    def build_raw(self, word_embedding, labels, sentences):
        self.word_embedding = word_embedding
        self.weights = {}
        words_in_categories = {}

        for i, label in enumerate(labels):
            for word in sentences[i]:
                if word not in words_in_categories:
                    words_in_categories[word] = {}
                if label not in words_in_categories[word]:
                    words_in_categories[word][label] = 1
                else:
                    words_in_categories[word][label] += 1

        for word, wordDict in words_in_categories.iteritems():
            counts = list(wordDict.itervalues())

            avg = sum(counts) / len(counts)
            sqr_avg = sum(i ** 2 for i in counts) / len(counts)
            deviation = max(sqr_avg - avg ** 2, 1) ** 0.5
            self.weights[word] = deviation

        self.sorted_words = sorted(self.weights.iteritems(), key=operator.itemgetter(1))
