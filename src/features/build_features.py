"""
Contains class FeatureBuilder for building feature set from given data set and word embedding
"""

import numpy as np
from src.data.make_dataset import *
from src.features.word_embeddings.iword_embedding import IWordEmbedding


class FeatureBuilder(object):
    """
    Class used for building feature matrix.
    Field "labels" is a list of categories of sentences
    Field "features" is a features matrix of shape (training set sixe, target_vector_length)
    """
    def __init__(self, embedding, labels, sentences):
        """
        :param embedding: instance of word embedding class implementing IWordEmbedding interface
        :param labels: list of labels of sentences
        :param sentences: list of sentences (as lists of words)
        """

        print ("Building features...")
        training_set_size = len(labels)
        self.labels = labels
        self.features = []

        for i in xrange(training_set_size):
            self.features.append(embedding.sentence_to_vector(sentences[i]))

    @staticmethod
    def get_features_path(data_folder):
        return os.path.join(os.path.dirname(__file__), '..\\..\\models\\features\\{0}_features.txt'.format(data_folder))

    def save(self, data_folder):
        """
        Saves features set in human-readable format in "models" folder
        """
        output_path = self.get_features_path(data_folder)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            for i, label in enumerate(self.labels):
                f.write("{0} {1}\n".format(label, ','.join(map(lambda val: str(val), self.features[i]))))


if __name__ == '__main__':
    """
    Main method will be for testing if FeatureBuilder works properly
    """

    from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding

    data_folder = "dataset1"
    data_file_path = get_external_data_path(data_folder)
    labels, sentences = read_dataset(data_folder)
    embedding = Word2VecEmbedding()

    if embedding.saved_embedding_exists(data_folder):
        print ("Using existing word embedding.")
        sentences = IWordEmbedding.data_file_to_sentences(data_file_path)
        embedding.load(data_folder, sentences)
    else:
        print ("Building word embedding...")
        embedding.build(sentences)

    fb = FeatureBuilder(embedding, labels, sentences )
    fb.save(data_folder)
    print "Processed features saved to " + fb.get_features_path(data_folder)
    print "{0} Labeled sentences".format(len(fb.labels))
    print "Labels: " + str(fb.labels)
    print "Features matrix shape: " + str(fb.features.shape)
