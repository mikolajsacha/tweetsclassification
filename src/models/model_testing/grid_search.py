""" Methods for performing grid search on model for a collection of parameters' combinations """
import ast
import os
import operator
from sklearn.model_selection import GridSearchCV
from src.common import DATA_FOLDER, CLASSIFIERS_PARAMS, SENTENCE_EMBEDDINGS, FOLDS_COUNT, LABELS, SENTENCES, \
    CLASSIFIERS_WRAPPERS, WORD_EMBEDDINGS
from src.features.build_features import FeatureBuilder

# don't remove this imports, they are needed to do eval() on string read from grid search result file
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings.sentence_embeddings import SumEmbedding, ConcatenationEmbedding


def get_grid_search_results_path(data_folder, classifier):
    name = classifier if isinstance(classifier, basestring) else classifier.__name__
    return os.path.join(os.path.dirname(__file__),
                        '../../../summaries/{0}_{1}_grid_search_results.txt'.format(data_folder, name))


def get_best_from_grid_search_results(classifier = None):
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        return None

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to get some results."
        return None

    max_result = 0.0
    best_parameters = None

    print("Found Grid Search results in " + summary_file_path.split("..")[-1])
    for line in open(summary_file_path, 'r'):
        split_line = tuple(line.split(";"))
        result = float(split_line[-1])
        if result > max_result:
            max_result = result
            best_parameters = split_line[:-1]

    word_embedding, word_embedding_params, sentence_embedding, classifier_params = best_parameters
    word_embedding_path, word_embedding_vector_length = tuple(word_embedding_params.split(','))
    # (word_emb_class, word_emb_params, sen_emb_class, clf_params)
    return eval(word_embedding), \
           [word_embedding_path, int(word_embedding_vector_length) ], \
           eval(sentence_embedding),\
           ast.literal_eval(classifier_params)


def grid_search(data_folder, folds_count, **kwargs):
    """ Performs grid search of all possible combinations of given parameters with logarithmic ranges.
        Saves results in formatted file in location pointed by get_grid_search_results_path method """

    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    classifiers = kwargs['classifiers']
    n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else 1

    # prepare output files
    for classifier_class, _ in classifiers:
        our_classifier_wrapper = CLASSIFIERS_WRAPPERS[classifier_class]
        output_path = get_grid_search_results_path(data_folder, our_classifier_wrapper)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        else: # clear output file
            with open(output_path, 'w'):
                pass

    for word_emb_class, word_emb_params in word_embeddings:
        word_embedding = word_emb_class(*word_emb_params)
        word_embedding.build()
        for sen_emb_class in sentence_embeddings:
            sen_emb = sen_emb_class()
            feature_builder = FeatureBuilder()
            str_word_emb_params = ','.join(map(str, word_emb_params))
            embedding_desc = ';'.join([word_emb_class.__name__, str_word_emb_params, sen_emb_class.__name__])
            print("Testing embedding: {0}".format(embedding_desc))

            sen_emb.build(word_embedding)
            feature_builder.build(sen_emb, LABELS, SENTENCES)

            for classifier_class, tested_params in classifiers:
                our_classifier_wrapper = CLASSIFIERS_WRAPPERS[classifier_class]
                output_path = get_grid_search_results_path(data_folder, our_classifier_wrapper)
                combs = reduce(operator.mul, map(len,tested_params.itervalues()) , 1)
                print("Testing {0} hyperparameters ({1} combinations)...".format(classifier_class.__name__, combs))

                clf = GridSearchCV(classifier_class(), tested_params, n_jobs=n_jobs)
                clf.fit(feature_builder.features, feature_builder.labels)

                with open(output_path, 'a') as output_file:
                    for mean_score, params in zip(clf.cv_results_['mean_test_score'], clf.cv_results_['params']):
                        output_file.write(
                            '{:s};{:s};{:4.2f}\n'.format(embedding_desc, str(params), mean_score*100))


if __name__ == "__main__":
    """ Runs grid search on a predefined set of parameters """

    classifiers_to_check = []
    for classifier_class, params in CLASSIFIERS_PARAMS:
        to_run = raw_input("Do you wish to test {0} with parameters {1} ? [y/n] "
                           .format(classifier_class.__name__, str(params)))
        if to_run.lower() == 'y' or to_run.lower() == 'yes':
            classifiers_to_check.append((classifier_class, params))

    print("*" * 20 + "\n")
    grid_search(DATA_FOLDER, FOLDS_COUNT,
                word_embeddings=WORD_EMBEDDINGS,
                sentence_embeddings=SENTENCE_EMBEDDINGS,
                classifiers=classifiers_to_check,
                n_jobs=-1)

