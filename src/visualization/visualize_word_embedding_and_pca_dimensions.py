import ast
import time

import multiprocessing
from numpy import NaN

from src.features.sentence_embeddings.sentence_embeddings import *
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_best_from_grid_search_results
from src.models.model_testing.validation import test_cross_validation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.visualization.save_visualization import save_current_plot
from src.common import DATA_FOLDER, choose_classifier, LABELS, SENTENCES
import warnings


def get_dim_results_path(data_folder, classifier):
    return os.path.join(os.path.dirname(__file__),
                        '../../summaries/{0}_{1}_word_emb_with_pca_dim_comparison_results.txt'
                        .format(data_folder, classifier.__name__))

if __name__ == "__main__":
    folds_count = 5
    dim_lengths = []
    dim_accuracies = []
    dim_execution_times = []
    dimensions = [1, 2, 3, 5, 10, 15, 25, 35, 45, 55, 70, 85, 100]

    classifier = choose_classifier()

    output_path = get_dim_results_path(DATA_FOLDER, classifier)

    use_new_dim = False

    if not (os.path.exists(output_path) and os.path.isfile(output_path)) or os.stat(output_path).st_size == 0:
        print "Results summary file does not exist or is empty. New search will be run."
        use_new_dim = True
    else:
        answer = raw_input("Do you wish to use existing results summary file [y] or run new calculations? [n] ")
        if answer.lower() == 'y' or answer.lower == 'yes':
            print ("Using existing PCA summary file...")

            for line in open(output_path, 'r'):
                word_emb_dim, pca_dim, dim_result, dim_execution_time = tuple(line.split(";"))
                dim_lengths.append((int(word_emb_dim), int(pca_dim)))
                dim_accuracies.append(float(dim_result))
                dim_execution_times.append(float(dim_execution_time))
        else:
            use_new_dim = True

    if use_new_dim:
        best_parameters = get_best_from_grid_search_results(classifier)

        if best_parameters is None:
            exit(-1)

        embedding, params = best_parameters
        word_emb_class, sen_emb_class = tuple(embedding.split(","))

        print ("\nEvaluating model for embedding {:s} with params {:s}".format(embedding, str(params)))
        params["n_jobs"] = multiprocessing.cpu_count()

        for word_emb_dim, pca_dim in dim_lengths:
            print ("\nCalculating model with Word Embeddings dimensions count = {0} ".format(word_emb_dim) +
                   "and PCA dimensions count = {0}".format(pca_dim))

            word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"), word_emb_dim)
            word_emb.build(SENTENCES)

            fb = build_features.FeatureBuilder()

            sen_emb = eval(sen_emb_class)(pca_dim)

            start_time = time.time()
            validation_results = test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb, fb,
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
            for i, (dim1, dim2) in enumerate(dim_lengths):
                output_file.write('{0};{1};{2};{3}\n'.format(dim1, dim2, dim_accuracies[i], dim_execution_times[i]))

    accuracy_legend = mpatches.Patch(color='b', label="Accuracy (as compared to the best)")
    execution_time_legend = mpatches.Patch(color='r', label="Execution time (as compared to the slowest)")

    fig = plt.figure(figsize=(12, 11))
    fig.suptitle('How number of dimensions in Word Embedding and PCA affects model accuracy (using {0})?'
                 .format(classifier.__name__))
    ax = plt.subplot(projection='3d')
    ax.set_xlabel('Number of Word Embedding dimensions')
    ax.set_ylabel('Number of PCA dimensions')
    ax.set_zlabel('Accuracy vs execution time')

    xs = []
    for i in xrange(len(dimensions)):
        xs.append([dimensions[i]] * (len(dimensions) - i))

    ys = []
    for i in xrange(len(dimensions)):
        ys.append(dimensions[i:])

    xs, ys = np.meshgrid(dimensions, dimensions)
    acc_zs = np.empty((len(dimensions), len(dimensions)), dtype=float)
    ex_zs = np.empty((len(dimensions), len(dimensions)), dtype=float)
    for i in xrange(len(dimensions)):
        for j in xrange(len(dimensions)):
            if ys[i, j] > xs[i, j]:
                acc_zs[i, j] = NaN
                ex_zs[i, j] = NaN
            else:
                ind = dim_lengths.index((xs[i, j], ys[i, j]))
                acc_zs[i, j] = dim_accuracies[ind]
                ex_zs[i, j] = dim_execution_times[ind]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.plot_surface(xs, ys, acc_zs, color='b', rstride=1, cstride=1, alpha=0.5)
        ax.plot_surface(xs, ys, ex_zs, color='r', rstride=1, cstride=1, alpha=0.5)

    plt.legend(handles=[accuracy_legend, execution_time_legend])
    save_current_plot('word_embedding_pca_dimensions.svg'.format(classifier.__name__))
    plt.show()
