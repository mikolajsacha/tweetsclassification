""" Methods for performing grid search on model for a collection of parameters' combinations """
import ast
import itertools
import os
import numpy as np
import multiprocessing
from multiprocessing.pool import Pool

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR

from src.data import make_dataset
from src.common import DATA_FOLDER, PROCESSED_DATA_PATH, EXTERNAL_DATA_PATH, \
    FOLDS_COUNT, SENTENCES, LABELS, CATEGORIES, log_range
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.research.word_categories import get_words_scores


def get_grid_search_results_path(data_folder, category):
    path = os.path.join(os.path.dirname(__file__),
                        '../../summaries/{0}_OwnAlgorithm_{1}_grid_search_results.txt'
                        .format(data_folder, category))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


def single_fold_validation(params):
    vectors = params['vectors']
    values = params['values']
    regressor_class = params['regressor_class']
    train_index = params['train_index']
    test_index = params['test_index']
    weighted = params['weighted']

    del params['vectors']
    del params['values']
    del params['regressor_class']
    del params['train_index']
    del params['test_index']
    del params['weighted']

    cls = regressor_class(**params)
    if weighted:
        cls.fit(vectors[train_index], values[train_index], values[train_index])
        return cls.score(vectors[test_index], values[test_index], values[test_index])
    else:
        cls.fit(vectors[train_index], values[train_index])
        return cls.score(vectors[test_index], values[test_index])


def grid_search(folds_count):
    classifier_params = [(SVR, {"C": list(log_range(-2, 12)), "gamma": list(log_range(-2, 10))}),
                         (RandomForestRegressor, {"n_estimators": [5,10,15,20,50,100,150,200,250,300]})]

    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    words_scores = get_words_scores(LABELS, SENTENCES)

    t_pool = Pool(multiprocessing.cpu_count())

    for cat_index, category in enumerate(CATEGORIES):
        print("." * 20)
        print("Testing for category {0}...".format(category))
        output_path = get_grid_search_results_path(DATA_FOLDER, category)
        # clear output file
        with open(output_path, 'w'):
            pass

        for regressor, tested_params in classifier_params:
            print("." * 20)
            print("Testing regressor: {0}...".format(regressor.__name__))

            params_values = ([(param, val) for val in values] for param, values in tested_params.iteritems())
            all_combinations = [dict(tuple_list) for tuple_list in itertools.product(*params_values)]


            skf = StratifiedKFold(n_splits=folds_count)
            fold = 0
            results_in_folds = [{}] * folds_count

            for train_index, test_index in skf.split(SENTENCES, LABELS):
                print("." * 20)
                print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
                print("Building embeddings...")
                training_sentences = SENTENCES[train_index]

                word_emb.build(training_sentences)

                word_vectors = np.array([word_emb[word] for word in words_scores.iterkeys()])
                values = np.array([v[cat_index] for v in words_scores.itervalues()])

                for params in all_combinations:
                    params['vectors'] = word_vectors
                    params['values'] = values
                    params['regressor_class'] = regressor
                    params['test_index'] = test_index
                    params['train_index'] = train_index
                    params['weighted'] = True

                print("Testing all hyperparameters combinations in fold {0}...".format(fold + 1))
                all_results_for_fold = t_pool.map(single_fold_validation, all_combinations)

                for params in all_combinations:
                    del params['vectors']
                    del params['values']
                    del params['regressor_class']
                    del params['test_index']
                    del params['train_index']
                    del params['weighted']

                for i in xrange(len(all_combinations)):
                    result = all_results_for_fold[i]
                    results_in_folds[fold][i] = result
                fold += 1

            with open(output_path, 'a') as output_file:
                for i in xrange(len(all_combinations)):
                    evaluation = sum(fold_results[i] for fold_results in results_in_folds) \
                                 / float(folds_count)
                    params = all_combinations[i]
                    output_file.write(
                        '{:s};{:s};{:f}\n'.format(regressor.__name__, str(params), evaluation))

            print "Results saved!"


def get_best_from_grid_search_results(category, verbose=True):
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, category)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        return None

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to get some results."
        return None

    max_result = -1000.0
    best_parameters = None

    if verbose:
        print("Found Grid Search results in " + summary_file_path.split("..")[-1])
    for line in open(summary_file_path, 'r'):
        regressor_class, params, result = tuple(line.split(";"))
        result = float(result)
        if result > max_result:
            max_result = result
            best_parameters = eval(regressor_class), ast.literal_eval(params)
    return best_parameters


def get_best_words_categories_regressors(word_scores, word_vectors, word_emb, verbose=False):
    categories_classifiers = []

    for cat_index, category in enumerate(CATEGORIES):
        regressor_class, params = get_best_from_grid_search_results(category, verbose=verbose)
        if verbose:
            print("Use {0}, {1}, for category {2}") \
                .format(regressor_class.__name__, str(params), category)

        clf = regressor_class(**params)
        values = np.array([v[cat_index] for v in word_scores.itervalues()])

        clf.fit(word_vectors, values, values)
        categories_classifiers.append(clf)
    return categories_classifiers


if __name__ == "__main__":
    """ Runs grid search on a predefined set of parameters """

    if not os.path.isfile(EXTERNAL_DATA_PATH):
        print "Path {0} does not exist".format(EXTERNAL_DATA_PATH)
        exit(-1)
    else:
        make_dataset.make_dataset(EXTERNAL_DATA_PATH, PROCESSED_DATA_PATH)

    grid_search(FOLDS_COUNT)
