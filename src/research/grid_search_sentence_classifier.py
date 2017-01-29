import ast
from multiprocessing import Pool
import multiprocessing
import numpy as np
import itertools
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.common import LABELS, CATEGORIES_COUNT, log_range, DATA_FOLDER, EXTERNAL_DATA_PATH, PROCESSED_DATA_PATH, \
    FOLDS_COUNT
from src.common import SENTENCES
from src.data import make_dataset
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.model_testing import grid_search as gs
from src.research.grid_search_word_classifier import get_best_words_categories_regressors, single_fold_validation
from src.research.word_categories import get_words_scores


def get_grid_search_results_path():
    return gs.get_grid_search_results_path(DATA_FOLDER, "OwnClassifier")


def grid_search(folds_count):
    print("Counting words' scores...")
    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    word_scores = get_words_scores(LABELS, SENTENCES)
    t_pool = Pool(multiprocessing.cpu_count())

    classifier_params = [(SVC, {"C": list(log_range(-15, 7, 2)), "gamma": list(log_range(-20, -5, 2))}),
                         (RandomForestClassifier, {"criterion": ["gini", "entropy"],
                                                   "n_estimators": [1,2,3,5,10,15,20,30],
                                                   "min_samples_split": [2, 10],
                                                   "min_samples_leaf": [1, 10],
                                                   "max_features": [None, "sqrt"]}),
                         (KNeighborsClassifier, {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]})
                         ]

    for i, (classifier_class, tested_params) in enumerate(classifier_params):
        params_values = ([(param, val) for val in values] for param, values in tested_params.iteritems())
        classifier_params[i] = (classifier_class,
                                [dict(tuple_list) for tuple_list in itertools.product(*params_values)])

    skf = StratifiedKFold(n_splits=folds_count)
    results_in_folds = [{}] * folds_count

    for fold, (train_index, test_index) in enumerate(skf.split(SENTENCES, LABELS)):
        print("." * 20)
        print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
        print("Building embeddings...")
        training_sentences = SENTENCES[train_index]

        word_emb.build(training_sentences)
        word_vectors = [word_emb[word] for word in word_scores.iterkeys()]

        print("Building word regressors...")
        word_regressors = get_best_words_categories_regressors(word_scores, word_vectors, word_emb, verbose=(fold==0))

        w_predictions = [cls.predict(word_vectors) for cls in word_regressors]
        w_indices = {}
        for i, word in enumerate(word_scores.iterkeys()):
            w_indices[word] = i

        features = np.zeros((len(SENTENCES), 4 * CATEGORIES_COUNT), dtype=float)
        for i, sentence in enumerate(SENTENCES):
            for j, cls in enumerate(word_regressors):
                predictions = [w_predictions[j][w_indices[word]] for word in sentence if word in word_emb.model]
                if not predictions:
                    continue
                pred_len = len(predictions)
                pred_avg = sum(predictions) / pred_len
                std_deviation = (sum(x ** 2 for x in predictions) / pred_len - (pred_avg / pred_len) ** 2) ** 0.5
                features[i][4 * j] = pred_avg
                features[i][4 * j + 1] = min(predictions)
                features[i][4 * j + 2] = max(predictions)
                features[i][4 * j + 3] = std_deviation

        for classifier_class, all_combinations in classifier_params:
            print("Testing {0}...".format(classifier_class.__name__))

            for params in all_combinations:
                params['vectors'] = features
                params['values'] = LABELS
                params['regressor_class'] = classifier_class
                params['test_index'] = test_index
                params['train_index'] = train_index
                params['weighted'] = False

            all_results_for_fold = t_pool.map(single_fold_validation, all_combinations)

            for params in all_combinations:
                del params['vectors']
                del params['values']
                del params['regressor_class']
                del params['test_index']
                del params['train_index']
                del params['weighted']

            results_in_folds[fold][classifier_class.__name__] = []
            for i in xrange(len(all_combinations)):
                result = all_results_for_fold[i]
                results_in_folds[fold][classifier_class.__name__].append(result)

    output_path = get_grid_search_results_path()
    with open(output_path, 'w') as output_file:
        for classifier_class, all_combinations in classifier_params:
            for i in xrange(len(all_combinations)):
                evaluation = sum(fold_results[classifier_class.__name__][i] for fold_results in results_in_folds) \
                             / float(folds_count) * 100
                params = all_combinations[i]
                output_file.write(
                    '{:s};{:s};{:f}\n'.format(classifier_class.__name__, str(params), evaluation))
    print "Results saved!"


def get_best_own_classifier_from_grid_search_results(features, labels):
    summary_file_path = get_grid_search_results_path()

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        return None

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to get some results."
        return None

    max_result = -1000.0
    best_parameters = None

    print("Found Grid Search results in " + summary_file_path.split("..")[-1])
    for line in open(summary_file_path, 'r'):
        classifier_class, params, result = tuple(line.split(";"))
        result = float(result)
        if result > max_result:
            max_result = result
            best_parameters = eval(classifier_class), ast.literal_eval(params)

    classifier_class, params = best_parameters
    print("Use {:s}, {:s} with score of {:4.4f}%").format(classifier_class.__name__, str(params), max_result)

    clf = classifier_class(**params)
    clf.fit(features, labels)
    return clf


if __name__ == "__main__":
    """ Runs grid search on a predefined set of parameters """

    if not os.path.isfile(EXTERNAL_DATA_PATH):
        print "Path {0} does not exist".format(EXTERNAL_DATA_PATH)
        exit(-1)
    else:
        make_dataset.make_dataset(EXTERNAL_DATA_PATH, PROCESSED_DATA_PATH)

    grid_search(FOLDS_COUNT)
