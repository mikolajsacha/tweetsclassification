"""
Contains methods for performing validation of learning models
"""
from sklearn.model_selection import cross_val_score
from src.data import make_dataset
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.features.build_features import FeatureBuilder
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def test_cross_validation(data_folder, word_embedding, sentence_embedding, feature_builder,
                          classifier_class, folds_count, **kwargs):
    data_file_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))

    labels, sentences = make_dataset.read_dataset(data_file_path, data_info)

    print("." * 20)
    print("Testing with {0}-fold cross-validation...".format(folds_count))

    # Uncomment for more verbose output
    # print("Building word embedding...")
    word_embedding.build(sentences)

    # print("Building sentence embedding...")
    sentence_embedding.build(word_embedding, labels, sentences)

    # print("Building classifier model...")
    classifier = classifier_class(sentence_embedding, **kwargs)

    # print("Building features...")
    feature_builder.build(sentence_embedding, labels, sentences)

    scores = cross_val_score(classifier.get_estimator(), feature_builder.features, feature_builder.labels,
                             cv=folds_count, n_jobs=-1)
    total_mean_result = scores.mean() * 100
    print("Mean cross-validation result: {:4.2f}% (+/- {:4.2f}%)".format(total_mean_result, scores.std() * 2 * 100))
    print("." * 20)

    return total_mean_result


def test_with_self(data_folder, word_embedding, sentence_embedding, feature_builder, classifier_class, **kwargs):
    # train for whole training set and check accuracy of prediction on it
    data_file_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    labels, sentences = make_dataset.read_dataset(data_file_path, data_info)

    print("." * 20)
    print("Testing predictions on the training set...")

    # uncomment prints if more verbose comments are preferred

    # print("Building word embedding...")
    word_embedding.build(sentences)

    # print("Building sentence embedding...")
    sentence_embedding.build(word_embedding, labels, sentences)

    # print("Building features...")
    feature_builder.build(sentence_embedding, labels, sentences)

    # print("Building classifier model...")
    classifier = classifier_class(sentence_embedding, **kwargs)
    classifier.fit(feature_builder.features, feature_builder.labels)

    # print("Making predictions...")
    successes = 0
    for i, label in enumerate(labels):
        prediction = classifier.predict(sentences[i])
        if int(prediction) == int(label):
            successes += 1

    set_length = len(labels)
    mean_result = successes / (1.0 * set_length) * 100
    print("Results when testing on training set: {0}/{1} successes ({2}%)"
          .format(successes, set_length, mean_result))
    return mean_result


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

    # train on the whole training set and test on the training set (just for curiosity)
    # test_with_self(data_folder, word_embedding, sentence_embedding, feature_builder, classifier, C=c)
    test_cross_validation(data_folder, word_embedding, sentence_embedding, feature_builder,
                          classifier, folds_count, C=c, gamma=gamma)
