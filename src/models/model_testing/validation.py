"""
Contains methods for performing validation of learning models
"""
import time
from sklearn.model_selection import StratifiedKFold

from src.common import LABELS, FOLDS_COUNT
from src.common import SENTENCES
from src.data import dataset
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.features.build_features import FeatureBuilder
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def single_fold_validation(training_features, training_labels, test_sentences, test_labels,
                           classifier_class, sentence_embedding, **kwargs):
    include_wrong_sentences = "include_wrong_sentences" in kwargs and kwargs["include_wrong_sentences"]
    if "include_wrong_sentences" in kwargs: del kwargs["include_wrong_sentences"]

    # test accuracy on a single fold with already built embeddings
    classifier = classifier_class(sentence_embedding, **kwargs)
    classifier.fit(training_features, training_labels)

    successes = 0
    wrong_sentences = []

    for i, label in enumerate(test_labels):
        prediction = classifier.predict(test_sentences[i])
        if int(prediction) == int(label):
            successes += 1
        elif include_wrong_sentences:
            wrong_sentences.append((' '.join(test_sentences[i]), label, prediction))

    ratio = float(successes) / len(test_labels)
    if include_wrong_sentences:
        return ratio, wrong_sentences
    return ratio


def test_cross_validation(labels, sentences, word_embedding, sentence_embedding,
                          classifier_class, folds_count, verbose=False, measure_times=False, **kwargs):
    # test accuracy for all folds combinations

    skf = StratifiedKFold(n_splits=folds_count)
    validation_results = []
    include_wrong_sentences = "include_wrong_sentences" in kwargs and kwargs["include_wrong_sentences"]
    if "include_wrong_sentences" in kwargs: del kwargs["include_wrong_sentences"]

    if verbose:
        print("Building word embedding...")
    word_embedding.build()

    fb = FeatureBuilder()

    training_time = 0
    testing_time = 0
    start_time = None

    if not sentence_embedding.use_pca:
        if measure_times: # measure training time -------------------
            start_time = time.time()
        if verbose:
            print("Building sentence embedding...")
        sentence_embedding.build(word_embedding)
        if verbose:
            print("Building features...")
        fb.build(sentence_embedding, labels, sentences)
        if measure_times: # measure training time -------------------
            # to be fair, multiply time with number of folds, because with PCA we will
            # have to build features folds_count times for cross-validation
            training_time += folds_count * (time.time() - start_time)

    for fold, (train_index, test_index) in enumerate(skf.split(sentences, labels)):
        if measure_times: # measure training time -------------------
            start_time = time.time()
        if verbose:
            print("Testing fold {0}/{1}...".format(fold + 1, folds_count))

        if sentence_embedding.use_pca:
            if verbose:
                print("Building sentence embedding...")
            sentence_embedding.build(word_embedding, sentences[train_index])
            if verbose:
                print("Building features...")
            fb = FeatureBuilder()
            fb.build(sentence_embedding, labels, sentences)

        if verbose:
            print("Building classifier model and testing predictions...")
        classifier = classifier_class(sentence_embedding, **kwargs)

        classifier.fit(fb.features[train_index], fb.labels[train_index])
        if measure_times: # measure training time -------------------
            training_time += time.time() - start_time

        if measure_times: # measure testing time -------------------
            start_time = time.time()
        success_rate = classifier.clf.score(fb.features[test_index], fb.labels[test_index])
        if measure_times: # measure testing time -------------------
            testing_time += time.time() - start_time

        if verbose:
            rate = success_rate[0] if include_wrong_sentences else success_rate
            print("Result in fold {:d}: {:4.2f}%".format(fold + 1, rate * 100))
        validation_results.append(success_rate)

    if measure_times:
        return validation_results, training_time, testing_time
    return validation_results


if __name__ == "__main__":
    """ Example of how cross validation works"""
    word_embedding = Word2VecEmbedding('google/GoogleNews-vectors-negative300.bin', 300)
    sentence_embedding = sentence_embeddings.ConcatenationEmbedding()

    classifier = SvmAlgorithm

    c = 100
    gamma = 0.1

    print("Testing parameter C={0}, gamma={1}...".format(c, gamma))
    print("." * 20)

    results = test_cross_validation(LABELS, SENTENCES, word_embedding, sentence_embedding, classifier,
                                    FOLDS_COUNT, True, C=c, gamma=gamma)
    print results
