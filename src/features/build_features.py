"""
Contains class FeatureBuilder for building feature set from given data set and word embedding
"""

from src.data.make_dataset import *
import numpy as np
import preprocessing


class FeatureBuilder(object):
    embedding_vector_length = 100

    def __init__(self, data_folder, embedding):
        input_file_path = get_external_data_path(data_folder)
        data_file_path = get_processed_data_path(data_folder)
        max_length = get_max_sentence_length(data_folder)

        make_dataset(input_file_path, data_file_path, max_length)
        embedding.build_from_data_set(data_folder, FeatureBuilder.embedding_vector_length)

        training_set_size = sum(1 for _ in open(data_file_path, 'r'))
        self.labels = [None] * training_set_size
        self.features = np.empty((training_set_size, FeatureBuilder.embedding_vector_length * max_length))

        with open(data_file_path, 'r') as f:
            for i, line in enumerate(f):
                label, sentence = line.split(' ', 1)
                words = sentence.rstrip().split(',')
                self.features[i] = np.concatenate(map(lambda w: embedding[w], words))
                self.labels[i] = label

        self.features = preprocessing.apply_pca(self.features)


if __name__ == '__main__':
    """
    Main method will be for testing if FeatureBuilder works properly
    """

    from src.features.word_embeddings.word_embeddings import Word2VecEmbedding

    fb = FeatureBuilder("dataset1", Word2VecEmbedding())
    print "{0} Labeled sentences".format(len(fb.labels))
    print "Labels: " + str(fb.labels)
    print "Features matrix shape: " + str(fb.features.shape)
