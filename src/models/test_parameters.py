from src.features.build_features import FeatureBuilder
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.models.algorithms import validation
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.data import make_dataset
import os


def test_parameters(data_folder, folds_count, classifier, sentence_embedding, word_embedding, **kwargs):
    feature_builder = FeatureBuilder()

    print("Testing parameters: {0}".format(str(kwargs)))

    return validation.test_cross_validation(data_folder, word_embedding, sentence_embedding,
                                            feature_builder, classifier, folds_count, **kwargs)


def test_all_params_combinations(param_name, data_folder, folds_count, **kwargs):
    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    classifiers = kwargs['classifiers']
    min_param_power = kwargs['min_param_power']
    max_param_power = kwargs['max_param_power']
    tested_params = kwargs['params']

    best_result = 0.0
    best_results_descriptions = None

    classifiers_results = []
    classifiers_params = []
    classifiers_descriptions = []

    for classifier in classifiers:
        for word_emb in word_embeddings:
            for sen_emb in sentence_embeddings:
                best_for_classifier_result = 0.0
                best_for_classifier_params = None
                classifier_desc = ','.join(map(lambda n: type(n).__name__, [classifier, word_emb, sen_emb]))
                classifiers_descriptions.append(classifier_desc)
                for params in []: # TODO: znajdz sposob na wszystkie kombinacje parametrow jako lista/generator
                    print("." * 20)
                    print("Testing " + classifier_desc)
                    result = test_parameters(param_name, data_folder, folds_count, classifier,
                                             sen_emb, word_emb, min_param_power, max_param_power, **params)
                    if result > best_for_classifier_result:
                        best_for_classifier_result = result
                        best_for_classifier_params = [params]
                    elif result == best_for_classifier_result:
                        best_for_classifier_params.append(params)

                    result_desc = classifier_desc + ", params = " + params
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
        print "Best result for classifier: {:s} is {:4.2f}% is for parameterss: " \
            .format(classifiers_descriptions[i], classifiers_results[i])
        for params in xrange(len(classifiers_params[i])):
            print str(params)

    print("." * 20)
    print("." * 20)
    print ("Best cross-validation result is {:4.2f}% for the following cases:".format(best_result))
    print("." * 20)
    for desc in best_results_descriptions:
        print desc


if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    folds_count = 5
    gamma_powers_range = -3, -1

    input_file_path = make_dataset.get_external_data_path(data_folder)
    output_file_path = make_dataset.get_processed_data_path(data_folder)

    if not os.path.isfile(input_file_path):
        print "Path {0} does not exist".format(input_file_path)
        exit(-1)
    else:
        make_dataset.make_dataset(input_file_path, output_file_path)

    word_embeddings = [Word2VecEmbedding(TextCorpora.get_corpus("brown"))]
    sentence_embeddings = [sentence_embeddings.ConcatenationEmbedding(),
                           sentence_embeddings.SumEmbedding(),
                           sentence_embeddings.TermCategoryVarianceEmbedding(),
                           sentence_embeddings.TermFrequencyAverageEmbedding(),
                           sentence_embeddings.ReverseTermFrequencyAverageEmbedding()]

    classifiers = [SvmAlgorithm()]

    test_all_params_combinations("gamma", data_folder, folds_count, word_embeddings=word_embeddings,
                                 sentence_embeddings=sentence_embeddings, classifiers=classifiers,
                                 min_param_power=gamma_powers_range[0], max_param_power=gamma_powers_range[1],
                                 other_params={"C": 100})
