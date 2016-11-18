"""
Contains class for usage of SVM algorithm
"""
from sklearn import svm

from src.data import make_dataset
import src.features.build_features as build_features
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """

    def train(self, labels, features, embedding, **kwargs):
        self.embedding = embedding
        self.clf = svm.SVC(**kwargs)
        self.clf.fit(features, labels)

    def predict(self, sentence):
        return int(self.clf.predict([self.embedding.sentence_to_vector(sentence)])[0])


if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    data_folder = "dataset1"
    labels, sentences = make_dataset.read_dataset(data_folder)
    embedding = Word2VecEmbedding()

    if embedding.saved_embedding_exists(data_folder):
        print ("Using existing word embedding.")
        embedding.load(data_folder, sentences)
    else:
        print ("Building word embedding...")
        embedding.build(sentences)
        embedding.save(data_folder)

    fb = build_features.FeatureBuilder()
    fb.build(embedding, labels, sentences)
    sentence_length = build_features.get_max_sentence_length(data_folder)
    svmAlg = SvmAlgorithm()
    svmAlg.train(fb.labels, fb.features, embedding)
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = build_features.sentence_to_word_vector(command, sentence_length)
        print svmAlg.predict(sentence)
