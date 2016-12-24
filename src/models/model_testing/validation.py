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


def test_cross_validation_dict(decorated_params):
    """ This method is a wrapper above test_cross_validation to use single dict as parameter.
        It makes multithreading using Thread Pools easier"""
    training_labels = decorated_params['training_labels']
    training_sentences = decorated_params['training_sentences']
    word_emb = decorated_params['word_emb']
    sen_emb = decorated_params['sen_emb']
    classifier = decorated_params['classifier']
    folds_count = decorated_params['folds_count']
    del decorated_params['training_labels']
    del decorated_params['training_sentences']
    del decorated_params['word_emb']
    del decorated_params['sen_emb']
    del decorated_params['classifier']
    del decorated_params['folds_count']
    return test_cross_validation(training_labels,
                                 training_sentences, word_emb, sen_emb,
                                 FeatureBuilder(), classifier, folds_count, verbose=False, **decorated_params)


def test_cross_validation(labels, sentences, word_embedding, sentence_embedding, feature_builder,
                          classifier_class, folds_count, verbose=False, **kwargs):
    # test accuracy for all folds combinations
    validation_results = []

    skf = StratifiedKFold(n_splits=folds_count)
    fold = 0

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
            print("Building classifier model...")
        classifier = classifier_class(sentence_embedding, **kwargs)
        classifier.fit(feature_builder.features, feature_builder.labels)

        successes = 0

        if verbose:
            print("Making predictions...")
        for i, label in enumerate(test_labels):
            prediction = classifier.predict(test_sentences[i])
            if int(prediction) == int(label):
                successes += 1

        success_rate = float(successes) / len(test_labels)

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
