import ast
import time
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from src.models.model_testing.validation import test_cross_validation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.save_visualization import save_current_plot
from src.configuration import DATA_FOLDER


def get_word_emb_dim_results_path(DATA_FOLDER, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '..\\..\\summaries\\{0}_{1}_word_emb_dim_comparison_results.txt'.format(DATA_FOLDER,
                                                                                                classifier.__name__))


if __name__ == "__main__":
    folds_count = 5
    classifiers = [SvmAlgorithm, RandomForestAlgorithm, NeuralNetworkAlgorithm]
    dim_lengths = []
    dim_accuracies = []
    dim_execution_times = []

    number = "x"
    print("Choose a classifier to test by typing a number:")
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(classifiers))

    while True:
        try:
            number = int(raw_input())
            if len(classifiers) > number >= 0:
                break
            else:
                raise ValueError()
        except ValueError:
            print "Please insert a correct number"

    classifier = classifiers[number]

    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))

    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier)
    output_path = get_word_emb_dim_results_path(DATA_FOLDER, classifier)

    use_new_word_embedding_dim = False

    if not (os.path.exists(output_path) and os.path.isfile(output_path)) or os.stat(output_path).st_size == 0:
        print "Word Embeddings Results summary file does not exist or is empty. New search will be run."
        use_new_word_embedding_dim = True
    else:
        answer = raw_input("Do you wish to use existing results summary file [y] or run new calculations? [n] ")
        if answer.lower() == 'y' or answer.lower == 'yes':
            print ("Using existing PCA summary file...")

            for line in open(output_path, 'r'):
                dim_length, dim_result, dim_execution_time = tuple(line.split(";"))
                dim_lengths.append(int(dim_length))
                dim_accuracies.append(float(dim_result))
                dim_execution_times.append(float(dim_execution_time))
        else:
            use_new_word_embedding_dim = True

    if use_new_word_embedding_dim:
        if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
            print "Grid Search summary file does not exist. Please run grid_search.py at first."
            exit(-1)

        if os.stat(summary_file_path).st_size == 0:
            print "Grid Search summary file is empty. Please run grid_search.py to gest some results."
            exit(-1)

        dim_lengths = [1, 2, 3, 5, 10, 15, 25, 35, 45, 55, 70, 85, 100]

        max_result = 0.0
        best_parameters = []

        print("Found Grid Search results in " + summary_file_path.split("..")[-1])
        for line in open(summary_file_path, 'r'):
            embedding, params, result = tuple(line.split(";"))
            if result > max_result:
                max_result = result
                best_parameters = embedding, ast.literal_eval(params)

        labels, sentences = make_dataset.read_dataset(data_path, data_info)
        embedding, params = best_parameters
        word_emb_class, sen_emb_class = tuple(embedding.split(","))

        print ("\nEvaluating model for embedding {:s} with params {:s}".format(embedding, str(params)))

        for dim_length in dim_lengths:
            print ("\nCalculating model with Word Embeddings dimensions count = {0}".format(dim_length))

            word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"), dim_length)
            word_emb.build(sentences)

            fb = build_features.FeatureBuilder()

            sen_emb = eval(sen_emb_class)(None)

            start_time = time.time()
            validation_results = test_cross_validation(labels, sentences, word_emb, sen_emb, fb,
                                                       classifier, folds_count, **params)
            dim_execution_times.append((time.time() - start_time))

            dim_accuracies.append(sum(validation_results) / folds_count)

        highest_execution_time = max(dim_execution_times)
        highest_score = max(dim_accuracies)

        # normalize times and accuracies
        for i in xrange(len(dim_lengths)):
            dim_execution_times[i] /= highest_execution_time
            dim_accuracies[i] /= highest_score

        with open(output_path, 'w') as output_file:
            for i, dim_length in enumerate(dim_lengths):
                output_file.write('{0};{1};{2}\n'.format(dim_length, dim_accuracies[i], dim_execution_times[i]))

    accuracy_legend = mpatches.Patch(color='b', label="Accuracy (as compared to the best)")
    execution_time_legend = mpatches.Patch(color='r', label="Execution time (as compared to the slowest)")

    plt.rcParams["figure.figsize"] = [11, 8]
    plt.legend(handles=[accuracy_legend, execution_time_legend])
    lines = plt.plot(dim_lengths, dim_accuracies, 'b', dim_lengths, dim_execution_times, 'r',
                     dim_lengths, dim_accuracies, 'bo', dim_lengths, dim_execution_times, 'ro')
    plt.setp(lines, linewidth=2, markersize=8)

    plt.title('How number of dimensions in Word Embedding affects model accuracy (using {0})?'
              .format(classifier.__name__))
    plt.xlabel('Number of Word Embedding dimensions')
    plt.ylabel('Accuracy vs execution time')
    save_current_plot('word_embedding_dimensions.svg'.format(classifier.__name__))
    plt.show()
