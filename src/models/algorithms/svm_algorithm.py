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


if __name__ == '__main__':
    """
    Main method is for testing SVM algorithm
    """

    data_folder = "dataset3_reduced"
    data_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    labels, sentences = make_dataset.read_dataset(data_path, data_info)
    word_embedding = Word2VecEmbedding(TextCorpora.get_corpus("brown"))

    if word_embedding.saved_embedding_exists(data_folder):
        print ("Using existing word embedding.")
        word_embedding.load(word_embedding.get_embedding_model_path(data_folder), sentences)
    else:
        print ("Building word embedding...")
        word_embedding.build(sentences)
        word_embedding.save(word_embedding.get_embedding_model_path(data_folder))

    print ("Building sentence embedding...")
    sentence_embedding = TermFrequencyAverageEmbedding()
    sentence_embedding.build(word_embedding, labels, sentences)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sentence_embedding, labels, sentences)

    svmAlg = SvmAlgorithm(sentence_embedding, C=10, gamma=1, probability=True)
    svmAlg.fit(fb.features, fb.labels)
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = make_dataset.string_to_words_list(command)
        print map(lambda (i, prob): "{:s}: {:4.2f}%".format(data_info["Categories"][i], 100.0*prob),
                  enumerate(svmAlg.predict_proba(sentence)))
