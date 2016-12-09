"""
Contains class for usage of SVM algorithm
"""
from sklearn import svm

from src.features import build_features
from src.data import make_dataset
from src.features.sentence_embeddings.sentence_embeddings import ConcatenationEmbedding, TermFrequencyAverageEmbedding, \
    SumEmbedding
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """
    def __init__(self, sentence_embedding, **kwargs):
        IClassificationAlgorithm.__init__(self)
        self.sentence_embedding = sentence_embedding
        self.clf = svm.SVC(**kwargs)

    def fit(self, features, labels):
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])

    def predict_proba(self, sentence):
        return self.clf.predict_proba([self.sentence_embedding[sentence]])[0]
