from src.features.build_features import FeatureBuilder
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.models.algorithms import validation
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.data import make_dataset
import os


def test_c_parameters(data_folder, folds_count,  classifier, sentence_embedding,
                      word_embedding, min_c_power, max_c_power):
    best_cross_result = 0.0
    best_c_param = None

    feature_builder = FeatureBuilder()

    tested_c_params = [10 ** i for i in xrange(min_c_power, max_c_power)]
    cross_results = []

    # test for various C parameters:
    for c in tested_c_params:
        print("." * 20)
        print("Testing parameter C={0}...".format(c))

        cross_result = validation.test_cross_validation(data_folder, word_embedding, sentence_embedding,
                                                        feature_builder, classifier, folds_count, C=c)
        cross_results.append(cross_result)
        if cross_result > best_cross_result:
            best_cross_result = cross_result
            best_c_param = c

    print ("Mean cross-validation results: ")
    for i, c in enumerate(tested_c_params):
        print ("C = {:10.2f}: cross-validation result: {:4.2f}%"
               .format(c, cross_results[i]))
    print ("Best cross-validation result is {0}% with parameter C={1}".format(best_cross_result, best_c_param))
    return best_cross_result, best_c_param


def test_parameters(data_folder, folds_count, **kwargs):
    sentence_embeddings = kwargs['sentence_embeddings']
    word_embeddings = kwargs['word_embeddings']
    classifiers = kwargs['classifiers']
    min_c_power = kwargs['min_c_power']
    max_c_power = kwargs['max_c_power']

    results_descriptions = []
    best_result = 0.0
    best_results_descriptions = ""

    for classifier in classifiers:
        for word_emb in word_embeddings:
            for sen_emb in sentence_embeddings:
                params_desc = "{:10s}, {:10s}, {:10s}" \
                              .format(type(classifier).__name__, type(word_emb).__name__, type(sen_emb).__name__)
                print("." * 20)
                print("Testing " + params_desc)
                result, best_c = test_c_parameters(data_folder, folds_count, classifier,
                                                   sen_emb, word_emb, min_c_power, max_c_power)
                results_desc = "Best C: {:10.2f} with result: {:4.2f}%".format(best_c, result)
                results_descriptions.append(params_desc + ", " + results_desc)
                if result > best_result:
                    best_result = result
                    best_results_descriptions = [params_desc + ", C={:10.2f}".format(best_c)]
                elif result == best_result:
                    best_results_descriptions.append(params_desc + ", C={:10.2f}".format(best_c))

    print("." * 20)
    print("Overall results:")
    print("." * 20)
    for desc in results_descriptions:
        print desc

    print("." * 20)
    print("." * 20)
    print ("Best cross-validation result is {:4.2f}% for following cases:".format(best_result))
    print("." * 20)
    for desc in best_results_descriptions:
        print desc


if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    folds_count = 5
    c_powers_range = 3, 6

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

    test_parameters(data_folder, folds_count, word_embeddings=word_embeddings, sentence_embeddings=sentence_embeddings,
                    classifiers=classifiers, min_c_power=c_powers_range[0], max_c_power=c_powers_range[1])

