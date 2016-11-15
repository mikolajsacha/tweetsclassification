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
    def __init__(self, data_folder, embedding):
        """
        :param data_folder: name of folder with processed data (e.g. "dataset1")
        :param embedding: instance of word embedding class implementing IWordEmbedding interface
        """
        self.data_folder = data_folder
        input_file_path = get_external_data_path(data_folder)
        data_file_path = get_processed_data_path(data_folder)

        # extend all sentences to the length of the longest sentence
        max_length = get_max_sentence_length(data_folder)

        # process data set
        make_dataset(input_file_path, data_file_path, max_length)

        # build embedding on processed data se
        embedding.build_from_data_set(data_folder, IWordEmbedding.initial_vector_length)

        training_set_size = sum(1 for _ in open(data_file_path, 'r'))
        self.labels = [None] * training_set_size
        self.features = np.empty((training_set_size, max_length * IWordEmbedding.target_vector_length))

        # safe features using word embedding
        with open(data_file_path, 'r') as f:
            for i, line in enumerate(f):
                label, sentence = line.split(' ', 1)
                words = sentence.rstrip().split(',')
                self.features[i] = np.concatenate(map(lambda w: embedding[w], words))
                self.labels[i] = label

    def get_features_path(self):
        return os.path.join(os.path.dirname(__file__), '..\\..\\models\\features\\{0}_features.txt'.format(self.data_folder))

    def save(self):
        """
        Saves features set in human-readable format in "models" folder
        """
        output_path = self.get_features_path()
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            for i, label in enumerate(self.labels):
                f.write("{0} {1}\n".format(label, ','.join(map(lambda val: str(val), self.features[i]))))


if __name__ == '__main__':
    """
    Main method will be for testing if FeatureBuilder works properly
    """

    from src.features.word_embeddings.word_embeddings import Word2VecEmbedding

    fb = FeatureBuilder("dataset1", Word2VecEmbedding())
    fb.save()
    print "Model saved to " + fb.get_features_path()
    print "{0} Labeled sentences".format(len(fb.labels))
    print "Labels: " + str(fb.labels)
    print "Features matrix shape: " + str(fb.features.shape)
