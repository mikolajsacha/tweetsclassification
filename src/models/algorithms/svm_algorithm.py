"""
Contains class for usage of SVM algorithm
"""
from src.features.build_features import FeatureBuilder, get_data_set_info_path
from src.models.algorithms.iclassification_algorithm import IClassificationAlgorithm
from sklearn import svm


class SvmAlgorithm(IClassificationAlgorithm):
    """
    Class for building model using Support Vector Machine method
    """

    def __init__(self, labels, features, data_set_info_path):
        data_set_info = self.read_data_info(data_set_info_path)
        self.categories = data_set_info["Categories"]
        self.clf = svm.SVC()
        self.clf.fit(features, labels)

    def predict(self, sentence):
        pass

if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    from src.features.word_embeddings.word_embeddings import Word2VecEmbedding

    fb = FeatureBuilder("dataset1", Word2VecEmbedding())
    data_set_info_path = get_data_set_info_path("dataset1")

    svmAlg = SvmAlgorithm(fb.labels, fb.features, data_set_info_path)
