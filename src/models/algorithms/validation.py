"""
Contains method for performing validation of learning models
"""
import itertools
import numpy as np

from src.data import make_dataset
from src.features.build_features import FeatureBuilder, read_dataset
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def test_cross_validation(data_folder, embedding_class, features_builder_class,
                          classification_class, folds_count, **kwargs):
    labels, sentences = read_dataset(data_folder)
    folded_labels = np.array_split(np.array(labels, dtype=int), folds_count)
    folded_sentences = np.array_split(np.array(sentences, dtype=object), folds_count)

    s_length = make_dataset.get_max_sentence_length(data_folder)

    # test accuracy for all folds combination
    total_successes = 0
    print("." * 20)
    print("Testing with {0} - fold cross-validation...".format(folds_count))
    for fold in xrange(folds_count):
        # print("." * 20)
        print("Testing fold {0}/{1}...".format(fold + 1, folds_count))

        # uncomment prints if more verbose comments are preffered
        # print("Slicing data set...")
        training_labels = list(itertools.chain(*[folded_labels[i] for i in xrange(folds_count) if i != fold]))
        training_sentences = list(itertools.chain(*[folded_sentences[i] for i in xrange(folds_count) if i != fold]))

        test_labels = folded_labels[fold]
        test_sentences = folded_sentences[fold]
        # print("Building embedding...")

        embedding = embedding_class()
        embedding.build(training_sentences)

        fb = features_builder_class(embedding, training_labels, training_sentences)
        # print("Building classifier model...")
        classifier = classification_class(fb.labels, fb.features, embedding, s_length, **kwargs)

        # print("Making predictions...")
        successes = 0
        for i, label in enumerate(test_labels):
            prediction = classifier.predict(test_sentences[i])
            if int(prediction) == int(label):
                successes += 1

        set_length = len(test_labels)
        print("Results in fold {0}: {1}/{2} successes ({3}%)" \
              .format(fold + 1, successes, set_length, successes / (1.0 * set_length) * 100))

        total_successes += successes

    print("." * 20)
    total_set_length = len(labels)
    total_mean_result = total_successes / (1.0 * total_set_length) * 100
    print("Total mean result: {0}/{1} successes ({2}%)".format(total_successes, total_set_length, total_mean_result))
    print("." * 20)

    return total_mean_result


def test_with_self(data_folder, embedding_class, features_builder_class, classification_class, **kwargs):
    # train for whole training set and check accuracy of prediction on it
    labels, sentences = read_dataset(data_folder)
    s_length = make_dataset.get_max_sentence_length(data_folder)

    print("." * 20)
    print("Testing predictions on the training set...")

    # uncomment prints if more verbose comments are preffered
    # print("Building embedding...")
    embedding = embedding_class()
    embedding.build(sentences)

    feature_builder = features_builder_class(embedding, labels, sentences)
    # print("Building classifier model...")
    classifier = classification_class(feature_builder.labels, feature_builder.features, embedding, s_length, **kwargs)

    # print("Making predictions...")
    successes = 0
    for i, label in enumerate(labels):
        prediction = classifier.predict(sentences[i])
        if int(prediction) == int(label):
            successes += 1

    set_length = len(labels)
    mean_result = successes / (1.0 * set_length) * 100
    print("Results when testing on training set: {0}/{1} successes ({2}%)" \
          .format(successes, set_length, mean_result))
    return mean_result


if __name__ == "__main__":
    data_folder = "dataset1"
    folds_count = 5

    best_cross_result = 0.0
    best_c_param = None

    tested_c_params = [10 ** i for i in xrange(-2, 6)]
    cross_results = []
    self_results = []

    # test for various C parameters:
    for c in tested_c_params:
        print("." * 20)
        print("Testing parameter C={0}...".format(c))
        print("." * 20)

        # train on the whole training set and test on the training set (just for curiosity)
        self_result = test_with_self(data_folder, Word2VecEmbedding, FeatureBuilder, SvmAlgorithm, C=c)

        cross_result = test_cross_validation(data_folder, Word2VecEmbedding, FeatureBuilder, SvmAlgorithm, folds_count, \
                                             C=c)
        self_results.append(self_result)
        cross_results.append(cross_result)
        if cross_result > best_cross_result:
            best_cross_result = cross_result
            best_c_param = c

    print ("Results of testing training set on itself and mean cross-validation results: ")
    for i, c in enumerate(tested_c_params):
        print ("C = {:8d}: with self: {:4.2f}%, cross-validation: {:4.2f}%".format(c, self_results[i], cross_results[i]))
    print ("Best cross-validation result is {0}% with parameter C={1}".format(best_cross_result, best_c_param))
