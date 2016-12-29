import itertools
import os
import multiprocessing
from multiprocessing.pool import Pool
from sklearn.model_selection import StratifiedKFold
from src.data import make_dataset
from src.features.build_features import FeatureBuilder
from src.features.sentence_embeddings import sentence_embeddings
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing import validation


def log_range(min_ten_power, max_ten_power):
    return (10 ** i for i in xrange(min_ten_power, max_ten_power))


def get_grid_search_results_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '..\\..\\..\\summaries\\{0}_{1}_grid_search_results.txt'.format(data_folder,
                                                                                        classifier.__name__))


def full_grid_search(data_folder, classifier, folds_count, **kwargs):
    """ Performs grid search of all possible combinations of given parameters with logarithmic ranges.
        Saves results in formatted file in location pointed by get_grid_search_results_path method """

    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    tested_params = kwargs['params']

    data_file_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    labels, sentences = make_dataset.read_dataset(data_file_path, data_info)

    params_values = ([(param, val) for val in values] for param, values in tested_params.iteritems())
    all_combinations = list(map(lambda tuple_list: dict(tuple_list), itertools.product(*params_values)))
    print("." * 20)

    output_path = get_grid_search_results_path(data_folder, classifier)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # clear output file
    with open(output_path, 'w'):
        pass

    t_pool = Pool(multiprocessing.cpu_count())

    for word_emb in word_embeddings:
        for sen_emb in sentence_embeddings:
            embedding_desc = ', '.join(map(lambda n: type(n).__name__, [word_emb, sen_emb]))

            print("." * 20)
            print("Testing embedding: {0}...".format(embedding_desc))

            skf = StratifiedKFold(n_splits=folds_count)
            feature_builder = FeatureBuilder()
            fold = 0
            results_in_folds = [{}] * folds_count

            for train_index, test_index in skf.split(sentences, labels):
                print("." * 20)
                print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
                print("Building embeddings...")
                training_labels = labels[train_index]
                training_sentences = sentences[train_index]

                test_labels = labels[test_index]
                test_sentences = sentences[test_index]

                word_emb.build(training_sentences)
                sen_emb.build(word_emb, training_labels, training_sentences)
                feature_builder.build(sen_emb, training_labels, training_sentences)

                for params in all_combinations:
                    params['training_features'] = feature_builder.features
                    params['training_labels'] = training_labels
                    params['test_sentences'] = test_sentences
                    params['test_labels'] = test_labels
                    params['sentence_embedding'] = sen_emb
                    params['classifier_class'] = classifier

                print("Testing all hyperparameters combinations in fold {0}...".format(fold + 1))
                all_results_for_fold = t_pool.map(validation.single_fold_validation_dict, all_combinations)

                for params in all_combinations:
                    del params['training_features']
                    del params['training_labels']
                    del params['test_sentences']
                    del params['test_labels']
                    del params['sentence_embedding']
                    del params['classifier_class']

                for i in xrange(len(all_combinations)):
                    result = all_results_for_fold[i]
                    results_in_folds[fold][i] = result
                fold += 1

            with open(output_path, 'a') as output_file:
                for i in xrange(len(all_combinations)):
                    evaluation = sum(fold_results[i] for fold_results in results_in_folds) \
                                 / float(folds_count) * 100
                    params = all_combinations[i]
                    output_file.write(
                        '{:s};{:s};{:4.2f}\n'.format(embedding_desc, str(params), evaluation))

            print "Results saved!"


def double_validation_grid_search(data_folder, classifier, folds_count, training_set_fraction, **kwargs):
    """ Performs grid search of all possible combinations of given parameters with logarithmic ranges.
        Evaluates only for best parameters for folds.
        Saves results in formatted file in location pointed by get_grid_search_results_path method """

    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    tested_params = kwargs['params']

    data_file_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    labels, sentences = make_dataset.read_dataset(data_file_path, data_info)

    training_set_size = int(labels.size * training_set_fraction)
    training_labels = labels[:training_set_size]
    training_sentences = sentences[:training_set_size]

    params_values = ([(param, val) for val in values] for param, values in tested_params.iteritems())
    all_combinations = list(map(lambda tuple_list: dict(tuple_list), itertools.product(*params_values)))

    print("." * 20)

    output_path = get_grid_search_results_path(data_folder, classifier)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # clear output file
    with open(output_path, 'w'):
        pass

    t_pool = Pool(multiprocessing.cpu_count())

    for word_emb in word_embeddings:
        for sen_emb in sentence_embeddings:
            embedding_desc = ', '.join(map(lambda n: type(n).__name__, [word_emb, sen_emb]))

            best_results = [0.0] * folds_count
            best_results_descriptions = [""] * folds_count
            best_results_params = [{}] * folds_count

            print("." * 20)
            print("Testing embedding: {0}...".format(embedding_desc))

            skf = StratifiedKFold(n_splits=folds_count)
            feature_builder = FeatureBuilder()
            fold = 0
            for train_index, test_index in skf.split(training_sentences, training_labels):
                print("." * 20)
                print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
                print("Building embeddings...")
                fold_training_labels = training_labels[train_index]
                fold_training_sentences = training_sentences[train_index]

                test_labels = training_labels[test_index]
                test_sentences = training_sentences[test_index]

                word_emb.build(fold_training_sentences)
                sen_emb.build(word_emb, fold_training_labels, fold_training_sentences)
                feature_builder.build(sen_emb, fold_training_labels, fold_training_sentences)

                for params in all_combinations:
                    params['training_features'] = feature_builder.features
                    params['training_labels'] = fold_training_labels
                    params['test_sentences'] = test_sentences
                    params['test_labels'] = test_labels
                    params['sentence_embedding'] = sen_emb
                    params['classifier_class'] = classifier

                print("Testing all hyperparameters combinations...")
                all_results_for_fold = t_pool.map(validation.single_fold_validation_dict, all_combinations)

                for params in all_combinations:
                    del params['training_features']
                    del params['training_labels']
                    del params['test_sentences']
                    del params['test_labels']
                    del params['sentence_embedding']
                    del params['classifier_class']

                for i, params in enumerate(all_combinations):
                    result_desc = embedding_desc + ", params = " + str(params)
                    result = all_results_for_fold[i]
                    if result > best_results[fold]:
                        best_results[fold] = result
                        best_results_descriptions[fold] = [result_desc]
                        best_results_params[fold] = [params]
                    elif result == best_results[fold]:
                        best_results_descriptions[fold].append(result_desc)
                        best_results_params[fold].append(params)
                fold += 1

            for i in xrange(folds_count):
                print("." * 20)
                print ("Best result for {:s}, fold {:d} is {:4.2f}% for the following cases:"
                       .format(embedding_desc, i + 1, best_results[i] * 100))
                for desc in best_results_descriptions[i]:
                    print desc

            print("." * 20)
            print("." * 20)
            print ("Evaluating the estimate of out-of-model errors for best parameters...")
            print("." * 20)
            print("." * 20)

            best_params = []
            best_params_descriptions = []
            for i in xrange(len(best_results_params)):
                for j in xrange(len(best_results_params[i])):
                    if best_results_params[i][j] not in best_params:
                        best_params.append(best_results_params[i][j])
                        best_params_descriptions.append(best_results_descriptions[i][j])

            fold = 0
            fold_evaluations = []

            for train_index, test_index in skf.split(sentences, labels):
                print("." * 20)
                print("Testing fold {0}/{1}...".format(fold + 1, folds_count))
                print("Building embeddings...")
                training_labels = labels[train_index]
                training_sentences = sentences[train_index]
                test_labels = labels[test_index]
                test_sentences = sentences[test_index]

                word_emb.build(training_sentences)
                sen_emb.build(word_emb, training_labels, training_sentences)
                feature_builder.build(sen_emb, training_labels, training_sentences)

                for i, params in enumerate(best_params):
                    params['training_features'] = feature_builder.features
                    params['training_labels'] = training_labels
                    params['test_sentences'] = test_sentences
                    params['test_labels'] = test_labels
                    params['sentence_embedding'] = sen_emb
                    params['classifier_class'] = classifier

                fold_evaluations.append(t_pool.map(validation.single_fold_validation_dict, best_params))
                fold += 1

            evaluations = [0.0] * len(best_params)
            for j in xrange(len(best_params)):
                for i in xrange(len(fold_evaluations)):
                    evaluations[j] += fold_evaluations[i][j]
                evaluations[j] /= float(len(fold_evaluations))

            for i in xrange(len(best_params_descriptions)):
                print "Model evaluation for {:s}: {:4.2f}%".format(best_params_descriptions[i], evaluations[i] * 100)

            with open(output_path, 'a') as output_file:
                for i, evaluation in enumerate(evaluations):
                    output_file.write(
                        '{:s};{:s};{:4.2f}\n'.format(embedding_desc, str(best_params[i]), evaluation * 100))

            print "Results saved!"


if __name__ == "__main__":
    """ Runs grid search on a predefined set of parameters """

    data_folder = "gathered_dataset"
    folds_count = 5
    training_set_size = 0.80
    algorithms = [(SvmAlgorithm, {"C": list(log_range(0, 6)), "gamma": list(log_range(-3, 2))}),
                  (RandomForestAlgorithm, {"criterion": ["gini", "entropy"],
                                           "min_samples_split": [2, 10],
                                           "min_samples_leaf": [1, 10],
                                           "max_features": [None, "sqrt"]}),
                  (NeuralNetworkAlgorithm, {"alpha": list(log_range(-5, -2)),
                                            "learning_rate": ["constant", "invscaling", "adaptive"],
                                            "activation": ["identity", "logistic", "tanh", "relu"],
                                            "hidden_layer_sizes": [(100,), (200,), (100, 50), (100, 100)]})
                  ]

    word_embeddings = [Word2VecEmbedding(TextCorpora.get_corpus("brown"))]
    sentence_embeddings = [
        sentence_embeddings.SumEmbedding(),
        sentence_embeddings.TermFrequencyAverageEmbedding()
    ]

    input_file_path = make_dataset.get_external_data_path(data_folder)
    output_file_path = make_dataset.get_processed_data_path(data_folder)

    if not os.path.isfile(input_file_path):
        print "Path {0} does not exist".format(input_file_path)
        exit(-1)
    else:
        make_dataset.make_dataset(input_file_path, output_file_path)

    for algorithm, params in algorithms:
        to_run = raw_input("Do you wish to test {0} with parameters {1} ? [y/n] "
                           .format(algorithm.__name__, str(params)))
        if to_run.lower() == 'y' or to_run.lower() == 'yes':
            search_type = raw_input("Do you wish to evalute all parameters performance [y] (may take very long) " +
                                    "or just look for best evaluation [n] (double cross-validation)? [y/n] ")
            if search_type.lower() == 'y' or search_type.lower() == 'yes':
                full_grid_search(data_folder, algorithm, folds_count,
                                 word_embeddings=word_embeddings,
                                 sentence_embeddings=sentence_embeddings,
                                 params=params)
            else:
                double_validation_grid_search(data_folder, algorithm, folds_count, training_set_size,
                                              # full=full,
                                              word_embeddings=word_embeddings,
                                              sentence_embeddings=sentence_embeddings,
                                              params=params)
