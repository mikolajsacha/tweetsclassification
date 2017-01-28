import ast
import time

import multiprocessing

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.features.word_embeddings.keras_word_embedding import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path, get_best_from_grid_search_results
from src.models.model_testing.validation import test_cross_validation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.save_visualization import save_current_plot
from src.common import DATA_FOLDER, choose_classifier, LABELS, SENTENCES


def get_pca_results_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '../../summaries/{0}_{1}_pca_comparison_results.txt'.format(data_folder,
                                                                                    classifier.__name__))

if __name__ == "__main__":
    folds_count = 5
    classifiers = [SvmAlgorithm, RandomForestAlgorithm, NeuralNetworkAlgorithm]
    pca_lengths = []
    pca_accuracies = []
    pca_execution_times = []

    classifier = choose_classifier()

    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))

    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier)
    output_path = get_pca_results_path(DATA_FOLDER, classifier)

    use_new_pca = False

    if not (os.path.exists(output_path) and os.path.isfile(output_path)) or os.stat(output_path).st_size == 0:
        print "PCA Results summary file does not exist or is empty. New search will be run."
        use_new_pca = True
    else:
        answer = raw_input("Do you wish to use existing PCA results summary file [y] or run new calculations? [n] ")
        if answer.lower() == 'y' or answer.lower == 'yes':
            print ("Using existing PCA summary file...")

            for line in open(output_path, 'r'):
                pca_length, pca_result, pca_execution_time = tuple(line.split(";"))
                pca_lengths.append(int(pca_length))
                pca_accuracies.append(float(pca_result))
                pca_execution_times.append(float(pca_execution_time))
        else:
            use_new_pca = True

    if use_new_pca:
        best_parameters = get_best_from_grid_search_results(classifier)

        if best_parameters is None:
            exit(-1)

        embedding, params = best_parameters
        word_emb_class, sen_emb_class = tuple(embedding.split(","))
        params["n_jobs"] = multiprocessing.cpu_count()

        pca_lengths = [1, 2, 3, 5, 10, 15, 25, 35, 45, 55, 70, 85, 100]

        print ("\nEvaluating model for embedding {:s} with params {:s}\n".format(embedding, str(params)))

        word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"), 100)

        fb = build_features.FeatureBuilder()

        for pca_length in pca_lengths:
            print ("Calculating model for PCA with dimensions reduced to {0}..".format(pca_length))

            sen_emb = eval(sen_emb_class)(pca_length)

            start_time = time.time()
            validation_results = test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb,
                                                       build_features.FeatureBuilder(), classifier,
                                                       folds_count, **params)
            pca_execution_times.append((time.time() - start_time))

            pca_accuracies.append(sum(validation_results) / folds_count)

        highest_execution_time = max(pca_execution_times)
        highest_score = max(pca_accuracies)

        # normalize times and accuracies
        for i in xrange(len(pca_lengths)):
            pca_execution_times[i] /= highest_execution_time
            pca_accuracies[i] /= highest_score

        with open(output_path, 'w') as output_file:
            for i, pca_length in enumerate(pca_lengths):
                output_file.write('{0};{1};{2}\n'.format(pca_length, pca_accuracies[i], pca_execution_times[i]))

    accuracy_legend = mpatches.Patch(color='b', label="Accuracy (as compared to the best)")
    execution_time_legend = mpatches.Patch(color='r', label="Execution time (as compared to the slowest)")

    plt.rcParams["figure.figsize"] = [11, 8]
    plt.legend(handles=[accuracy_legend, execution_time_legend])
    lines = plt.plot(pca_lengths, pca_accuracies, 'b', pca_lengths, pca_execution_times, 'r',
                     pca_lengths, pca_accuracies, 'bo', pca_lengths, pca_execution_times, 'ro')
    plt.setp(lines, linewidth=2, markersize=8)

    plt.title('How PCA dimension reduction affects model accuracy (using {0})?'.format(classifier.__name__))
    plt.xlabel('PCA dimensions')
    plt.ylabel('Accuracy vs execution time')
    save_current_plot('pca_{0}.svg'.format(classifier.__name__))
    plt.show()
