from src.data import make_dataset
from src.features.build_features import FeatureBuilder
from src.models.algorithms import validation
import itertools
import os


def log_range(min_ten_power, max_ten_power):
    return (10 ** i for i in xrange(min_ten_power, max_ten_power))


def get_test_summary_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '..\\..\\summaries\\{0}_{1}.txt'.format(data_folder, classifier.__name__))


def test_all_params_combinations(data_folder, classifier, folds_count, training_set_fraction, **kwargs):
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
    all_combinations = map(lambda tuple_list: dict(tuple_list), itertools.product(*params_values))

    print("." * 20)

    output_path = get_test_summary_path(data_folder, classifier)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'w') as output_file:
        for word_emb in word_embeddings:
            for sen_emb in sentence_embeddings:
                embedding_desc = ', '.join(map(lambda n: type(n).__name__, [word_emb, sen_emb]))

                best_results = [0.0] * folds_count
                best_results_descriptions = [""] * folds_count
                best_results_params = [{}] * folds_count

                print("." * 20)
                print("Testing embedding: " + embedding_desc)
                for params in all_combinations:
                    result_desc = embedding_desc + ", params = " + str(params)
                    print("." * 20)
                    print("Testing " + result_desc)
                    results = validation.test_cross_validation(training_labels, training_sentences, word_emb, sen_emb,
                                                               FeatureBuilder(), classifier, folds_count, **params)
                    for i in xrange(folds_count):
                        result = results[i]
                        if result > best_results[i]:
                            best_results[i] = result
                            best_results_descriptions[i] = [result_desc]
                            best_results_params[i] = [params]
                        elif result == best_results[i]:
                            best_results_descriptions[i].append(result_desc)
                            best_results_params[i].append(params)

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
                already_checked_params = set()
                for i in xrange(folds_count):
                    zipped_params = zip(best_results_descriptions[i], best_results_params[i])
                    for desc, params, in zipped_params:
                        if desc not in already_checked_params:
                            already_checked_params.add(desc)
                            print("Testing " + desc)
                            results = validation.test_cross_validation(labels, sentences, word_emb, sen_emb,
                                                                       FeatureBuilder(), classifier,
                                                                       folds_count, **params)
                            mean = sum(results) / float(len(results)) * 100.0
                            print("." * 20)
                            print ("Model evaluation result for {:s} is {:4.2f}%"
                                   .format(desc, mean))
                            output_file.write('{:s};{:s};{:4.2f}\n'.format(embedding_desc, str(params), mean))
                            print("." * 20)
