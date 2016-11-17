"""
Contains class for usage of SVM algorithm
"""
from src.features.build_features import FeatureBuilder, sentence_to_word_vector, get_max_sentence_length
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn import svm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """

    def __init__(self, labels, features, embedding, sentence_length):
        self.embedding = embedding
        self.clf = svm.SVC()
        self.clf.fit(features, labels)
        self.sentence_length = sentence_length

    def predict(self, sentence):
        return int(self.clf.predict([self.embedding.sentence_to_vector(sentence)])[0])

if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding

    embedding = Word2VecEmbedding()
    fb = FeatureBuilder("dataset1", embedding)
    sentence_length = get_max_sentence_length("dataset1")

    svmAlg = SvmAlgorithm(fb.labels, fb.features, embedding, sentence_length)
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = sentence_to_word_vector(command, svmAlg.sentence_length)
        print svmAlg.predict(sentence)
