"""
Contains method for performing validation of learning models
"""
import itertools
import numpy as np

from src.data import make_dataset
from src.features.build_features import FeatureBuilder, read_dataset
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def test_cross_validation(data_folder, embedding_class, features_builder_class, classification_class, folds_count):
    labels, sentences = read_dataset(data_folder)
    folded_labels = np.array_split(np.array(labels, dtype=int), folds_count)
    folded_sentences = np.array_split(np.array(sentences, dtype=object), folds_count)

    s_length = make_dataset.get_max_sentence_length(data_folder)

    # test accuracy for all folds combination
    total_successes = 0
    for fold in xrange(folds_count):
        print("." * 20)
        print("Testing fold {0}/{1}".format(fold + 1, folds_count))

        training_labels = list(itertools.chain(*[folded_labels[i] for i in xrange(folds_count) if i != fold]))
        training_sentences = list(itertools.chain(*[folded_sentences[i] for i in xrange(folds_count) if i != fold]))

        test_labels = folded_labels[fold]
        test_sentences = folded_sentences[fold]

        embedding = embedding_class()
        embedding.build(training_sentences)

        feature_builder = features_builder_class(embedding, training_labels, training_sentences)
        classifier = classification_class(feature_builder.labels, feature_builder.features, embedding, s_length)

        successes = 0
        for i, label in enumerate(test_labels):
            sentence = test_sentences[0]
            prediction = classifier.predict(sentence)
            if prediction == label:
                successes += 1

        set_length = len(test_labels)
        print("Results in fold {0}: {1}/{2} successes ({3}%)" \
              .format(fold + 1, successes, set_length, successes / (1.0 * set_length) * 100))

        total_successes += successes

    print("." * 20)
    total_set_length = len(labels)
    print("Total mean result: {0}/{1} successes ({2}%)" \
          .format(total_successes, total_set_length, total_successes / (1.0 * total_set_length) * 100))
    print("." * 20)


if __name__ == "__main__":
    data_folder = "dataset2"
    folds_count = 5

    test_cross_validation(data_folder, Word2VecEmbedding, FeatureBuilder, SvmAlgorithm, folds_count)
