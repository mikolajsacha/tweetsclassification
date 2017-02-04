import os
import time
import numpy as np

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path, get_best_from_grid_search_results
from src.models.model_testing.validation import test_cross_validation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.save_visualization import save_current_plot
from src.common import DATA_FOLDER, choose_classifier, LABELS, SENTENCES, FOLDS_COUNT, DATA_SIZE


def get_pca_results_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '../../summaries/{0}_{1}_pca_comparison_results.txt'.format(data_folder,
                                                                                    classifier.__name__))


def calculate_pca_accuracies(classifier_class, pca_count, n_jobs):
    output_path = get_pca_results_path(DATA_FOLDER, classifier_class)
    pca_execution_times = []
    pca_accuracies = []
    best_parameters = get_best_from_grid_search_results(classifier_class)

    if best_parameters is None:
        exit(-1)

    word_emb_class, word_emb_params, sen_emb_class, params = best_parameters


    print ("\nEvaluating model for word embedding: {:s}({:s}), sentence embedding: {:s} \nHyperparameters {:s}\n"
           .format(word_emb_class.__name__, ', '.join(map(str, word_emb_params)), sen_emb_class.__name__, str(params)))
    params["n_jobs"] = n_jobs

    word_emb = word_emb_class(*word_emb_params)
    sen_emb = sen_emb_class()
    sen_emb.build(word_emb)

    max_pca_dim = int(DATA_SIZE / float(FOLDS_COUNT) * (FOLDS_COUNT - 1)) # PCA dim can't be greater than samples count
    pca_lengths = map(int, np.linspace(1, min(max_pca_dim, sen_emb.word_vector_length), num=pca_count))

    for pca_length in pca_lengths:
        print ("Calculating model for PCA with dimensions reduced to {0}..".format(pca_length))

        sen_emb = sen_emb_class(pca_length)

        start_time = time.time()
        validation_results = test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb, classifier_class,
                                                   FOLDS_COUNT, **params)
        pca_execution_times.append((time.time() - start_time))

        pca_accuracies.append(sum(validation_results) / FOLDS_COUNT)

    highest_execution_time = max(pca_execution_times)
    highest_score = max(pca_accuracies)

    # normalize times and accuracies
    for i in xrange(len(pca_lengths)):
        pca_execution_times[i] /= highest_execution_time
        pca_accuracies[i] /= highest_score

    with open(output_path, 'w') as output_file:
        for i, pca_length in enumerate(pca_lengths):
            output_file.write('{0};{1};{2}\n'.format(pca_length, pca_accuracies[i], pca_execution_times[i]))

    return pca_lengths, pca_accuracies, pca_execution_times

def visualize_pca(classifier_class, n_jobs):
    output_path = get_pca_results_path(DATA_FOLDER, classifier_class)
    pca_lengths = []
    pca_accuracies = []
    pca_execution_times = []

    use_new_pca = False

    if not (os.path.exists(output_path) and os.path.isfile(output_path)) or os.stat(output_path).st_size == 0:
        print "PCA Results summary file does not exist or is empty. New search will be run."
        use_new_pca = True
    else:
        answer = raw_input("Do you wish to use existing PCA results summary file [y] or run new calculations? [n] ")
        if answer.lower() == 'y' or answer.lower == 'yes':
            print ("Using existing PCA summary file...")
        else:
            use_new_pca = True

    if use_new_pca:
        pca_count = 10 # may be changed
        pca_lengths, pca_accuracies, pca_execution_times = calculate_pca_accuracies(classifier_class, pca_count, n_jobs)
    else:
        for line in open(output_path, 'r'):
            pca_length, pca_result, pca_execution_time = tuple(line.split(";"))
            pca_lengths.append(int(pca_length))
            pca_accuracies.append(float(pca_result))
            pca_execution_times.append(float(pca_execution_time))

    accuracy_legend = mpatches.Patch(color='b', label="Accuracy (as compared to the best)")
    execution_time_legend = mpatches.Patch(color='r', label="Execution time (as compared to the slowest)")

    plt.rcParams["figure.figsize"] = [11, 8]
    plt.legend(handles=[accuracy_legend, execution_time_legend])
    lines = plt.plot(pca_lengths, pca_accuracies, 'b', pca_lengths, pca_execution_times, 'r',
                     pca_lengths, pca_accuracies, 'bo', pca_lengths, pca_execution_times, 'ro')
    plt.setp(lines, linewidth=2, markersize=8)

    plt.title('How PCA dimension reduction affects model accuracy (using {0})?'.format(classifier_class.__name__))
    plt.xlabel('PCA dimensions')
    plt.ylabel('Accuracy vs execution time')
    save_current_plot('pca_{0}.svg'.format(classifier_class.__name__))
    plt.show()


if __name__ == "__main__":
    classifiers = [SvmAlgorithm, RandomForestAlgorithm, NeuralNetworkAlgorithm]
    classifier_class = choose_classifier()
    visualize_pca(classifier_class, n_jobs=-1)

