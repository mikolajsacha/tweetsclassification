"""
Contains class FeatureBuilder for building feature set from given data set and word embedding
"""

import numpy as np
from src.data.make_dataset import *
from src.features.sentence_embeddings.sentence_embeddings import ConcatenationEmbedding


class FeatureBuilder(object):
    """
    Class used for building feature matrix.
    Field "labels" is a list of categories of sentences
    Field "features" is a features matrix of shape (training set sixe, target_vector_length)
    """

    def __init__(self):
        self.labels = []
        self.features = []

    def build(self, sentence_embedding, labels, sentences):
        """
        :param sentence_embedding: instance of sentence embedding class implementing ISentenceEmbedding interface
        :param labels: list of labels of sentences
        :param sentences: list of sentences (as lists of words)
        """
        training_set_size = len(labels)
        self.labels = labels
        self.features = []

        for i in xrange(training_set_size):
            self.features.append(sentence_embedding[sentences[i]])

    def save(self, data_folder):
        """
        Saves features set in human-readable format in "models" folder
        """
        output_path = FeatureBuilder.get_features_path(data_folder)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            for i, label in enumerate(self.labels):
                f.write("{0} {1}\n".format(label, ','.join(map(lambda val: str(val), self.features[i]))))

    @staticmethod
    def get_features_path(data_folder):
        return os.path.join(os.path.dirname(__file__), '..\\..\\models\\features\\{0}_features.txt'.format(data_folder))


if __name__ == '__main__':
    """
    Main method will be for testing if FeatureBuilder works properly
    """

    from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding

    data_folder = "dataset1"
    data_file_path = get_processed_data_path(data_folder)
    data_info = read_data_info(get_data_set_info_path(data_folder))

    labels, sentences = read_dataset(data_file_path, data_info)
    word_embedding = Word2VecEmbedding()

    if word_embedding.saved_embedding_exists(data_folder):
        print ("Using existing word embedding.")
        word_embedding.load(word_embedding.get_embedding_model_path(data_folder), sentences)
    else:
        print ("Building word embedding...")
        word_embedding.build(sentences)
        print ("Saving word embedding...")
        word_embedding.save(word_embedding.get_embedding_model_path(data_folder))

    print ("Building sentence embedding...")
    sentence_embedding = ConcatenationEmbedding()
    sentence_embedding.build(word_embedding)

    print ("Building features...")
    fb = FeatureBuilder()
    fb.build(sentence_embedding, labels, sentences)
    fb.save(data_folder)
    print "Processed features saved to " + fb.get_features_path(data_folder)
    print "{0} Labeled sentences".format(len(fb.labels))
    print "Features matrix shape: {0} * {1}".format(len(fb.features), len(fb.features[0]))
