"""
Contains class for usage of SVM algorithm
"""
from src.features.build_features import FeatureBuilder, get_data_set_info_path, sentence_to_word_vector, \
    get_max_sentence_length
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn import svm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """

    def __init__(self, labels, features, embedding, data_set_info_path, sentence_length):
        data_set_info = self.read_data_info(data_set_info_path)
        self.embedding = embedding
        self.categories = data_set_info["Categories"]
        self.clf = svm.SVC()
        self.clf.fit(features, labels)
        self.sentence_length = sentence_length

    def predict(self, sentence):
        keywords = sentence_to_word_vector(sentence, self.sentence_length)
        return self.categories[int(self.clf.predict([self.embedding.sentence_to_vector(keywords)])[0])]

if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding

    embedding = Word2VecEmbedding()
    fb = FeatureBuilder("dataset1", embedding)
    data_set_info_path = get_data_set_info_path("dataset1")
    sentence_length = get_max_sentence_length("dataset1")

    svmAlg = SvmAlgorithm(fb.labels, fb.features, embedding, data_set_info_path, sentence_length)
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        print svmAlg.predict(command)
