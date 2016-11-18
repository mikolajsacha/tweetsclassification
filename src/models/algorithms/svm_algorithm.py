"""
Contains class for usage of SVM algorithm
"""
from sklearn import svm

import src.features.build_features as build_features
from src.data import make_dataset
from src.features.sentence_embeddings.sentence_embeddings import ConcatenationEmbedding
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """

    def train(self, labels, features, sentence_embedding, **kwargs):
        self.sentence_embedding = sentence_embedding
        self.clf = svm.SVC(**kwargs)
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.sentence_embedding[sentence]])[0])


if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    data_folder = "dataset1"
    labels, sentences = make_dataset.read_dataset(data_folder)
    word_embedding = Word2VecEmbedding()

    if word_embedding.saved_embedding_exists(data_folder):
        print ("Using existing word embedding.")
        word_embedding.load(data_folder, sentences)
    else:
        print ("Building word embedding...")
        word_embedding.build(sentences)
        word_embedding.save(data_folder)

    print ("Building sentence embedding...")
    sentence_embedding = ConcatenationEmbedding()
    sentence_embedding.build(word_embedding)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sentence_embedding, labels, sentences)

    sentence_length = build_features.get_max_sentence_length(data_folder)

    svmAlg = SvmAlgorithm()
    svmAlg.train(fb.labels, fb.features, sentence_embedding)
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = build_features.sentence_to_word_vector(command, sentence_length)
        print svmAlg.predict(sentence)
