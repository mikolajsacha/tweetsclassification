"""
Contains methods for performing validation of learning models
"""
from sklearn.model_selection import StratifiedKFold
from src.data import make_dataset
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.features.build_features import FeatureBuilder
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def single_fold_validation_dict(params):
    """ This method is a wrapper on single_fold_validation to use single dict as parameter.
        It makes multithreading using Thread Pools easier"""
    training_features = params['training_features']
    training_labels = params['training_labels']
    test_sentences = params['test_sentences']
    test_labels = params['test_labels']
    sentence_embedding = params['sentence_embedding']
    classifier_class = params['classifier_class']
    del params['training_features']
    del params['training_labels']
    del params['test_sentences']
    del params['test_labels']
    del params['sentence_embedding']
    del params['classifier_class']
    return single_fold_validation(training_features, training_labels, test_sentences, test_labels,
                                    classifier_class, sentence_embedding, **params)


def single_fold_validation(training_features, training_labels, test_sentences, test_labels,
                           classifier_class, sentence_embedding, **kwargs):
    # test accuracy on a single fold with already built embeddings
    classifier = classifier_class(sentence_embedding, **kwargs)
    classifier.fit(training_features, training_labels)

    successes = 0

    for i, label in enumerate(test_labels):
        prediction = classifier.predict(test_sentences[i])
        if int(prediction) == int(label):
            successes += 1

    return float(successes) / len(test_labels)


def test_cross_validation(labels, sentences, word_embedding, sentence_embedding, feature_builder,
                          classifier_class, folds_count, verbose=False, **kwargs):
    # test accuracy for all folds combinations

    skf = StratifiedKFold(n_splits=folds_count)
    fold = 0
    validation_results = []

    for train_index, test_index in skf.split(sentences, labels):
        if verbose:
            print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
            print("Slicing data set...")
        training_labels = labels[train_index]
        training_sentences = sentences[train_index]

        test_labels = labels[test_index]
        test_sentences = sentences[test_index]

        if verbose:
            print("Building word embedding...")
        word_embedding.build(training_sentences)

        if verbose:
            print("Building sentence embedding...")
        sentence_embedding.build(word_embedding, training_labels, training_sentences)

        if verbose:
            print("Building features...")
        feature_builder.build(sentence_embedding, training_labels, training_sentences)

        if verbose:
            print("Building classifier model and testing predictions...")
        success_rate = single_fold_validation(feature_builder.features, feature_builder.labels,
                                              test_sentences, test_labels,
                                              classifier_class, sentence_embedding, **kwargs)

        if verbose:
            print("Result in fold {:d}: {:4.2f}%".format(fold + 1, success_rate * 100))
        validation_results.append(success_rate)
        fold += 1

    return validation_results


if __name__ == "__main__":
    """ Example of how cross validation works"""
    data_folder = "dataset1"
    folds_count = 5

    word_embedding = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    sentence_embedding = sentence_embeddings.ConcatenationEmbedding()

    feature_builder = FeatureBuilder()
    classifier = SvmAlgorithm

    c = 100
    gamma = 0.1

    print("Testing parameter C={0}, gamma={1}...".format(c, gamma))
    print("." * 20)

    data_file_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    labels, sentences = make_dataset.read_dataset(data_file_path, data_info)

    test_cross_validation(labels, sentences, word_embedding, sentence_embedding, feature_builder,
                          classifier, folds_count, False, C=c, gamma=gamma)
