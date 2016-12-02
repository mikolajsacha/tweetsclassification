from src.features.build_features import FeatureBuilder
from src.models.algorithms import validation
import itertools
import os


def get_test_summary_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '..\\..\\models\\test_summaries\\{0}_{1}.txt'.format(data_folder, type(classifier).__name__))


def test_parameters(data_folder, folds_count, classifier, sentence_embedding, word_embedding, **kwargs):
    feature_builder = FeatureBuilder()

    print("Testing parameters: {0}".format(str(kwargs)))

    return validation.test_cross_validation(data_folder, word_embedding, sentence_embedding,
                                            feature_builder, classifier, folds_count, **kwargs)


def test_all_params_combinations(data_folder, classifier, folds_count, **kwargs):
    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    tested_params = kwargs['params']

    params_values = ([(param, val) for val in values] for param, values in tested_params.iteritems())
    all_combinations = map(lambda tuple_list: dict(tuple_list), itertools.product(*params_values))

    best_result = 0.0
    best_results_descriptions = None

    classifiers_results = []
    classifiers_params = []
    classifiers_descriptions = []

    output_path = get_test_summary_path(data_folder, classifier)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, 'w') as output_file:
        for word_emb in word_embeddings:
            for sen_emb in sentence_embeddings:
                best_for_classifier_result = 0.0
                best_for_classifier_params = None
                classifier_desc = ', '.join(map(lambda n: type(n).__name__, [word_emb, sen_emb]))
                classifiers_descriptions.append(classifier_desc)
                for params in all_combinations:
                    print("." * 20)
                    print("Testing " + classifier_desc)
                    result = test_parameters(data_folder, folds_count, classifier, sen_emb, word_emb, **params)
                    output_file.write('{:s};{:s};{:4.2f}\n'.format(classifier_desc, str(params), result))
                    if result > best_for_classifier_result:
                        best_for_classifier_result = result
                        best_for_classifier_params = [params]
                    elif result == best_for_classifier_result:
                        best_for_classifier_params.append(params)

                    result_desc = classifier_desc + ", params = " + str(params)
                    if result > best_result:
                        best_result = result
                        best_results_descriptions = [result_desc]
                    elif result == best_result:
                        best_results_descriptions.append(result_desc)

                classifiers_results.append(best_for_classifier_result)
                classifiers_params.append(best_for_classifier_params)

    print("." * 20)
    print("Overall results:")
    print("." * 20)
    for i in xrange(len(classifiers_results)):
        print "Best result for classifier: {:s} is {:4.2f}% is for parameters: " \
            .format(classifiers_descriptions[i], classifiers_results[i])
        for params in classifiers_params[i]:
            print str(params)

    print("." * 20)
    print("." * 20)
    print ("Overall best cross-validation result is {:4.2f}% for the following cases:".format(best_result))
    print("." * 20)
    for desc in best_results_descriptions:
        print desc


def log_range(min_ten_power, max_ten_power):
    return (10 ** i for i in xrange(min_ten_power, max_ten_power))
