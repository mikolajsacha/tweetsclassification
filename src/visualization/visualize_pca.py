import os

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_best_from_grid_search_results
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
    training_times = []
    testing_times = []
    pca_accuracies = []
    best_parameters = get_best_from_grid_search_results(classifier_class)

    if best_parameters is None:
        exit(-1)

    word_emb_class, word_emb_params, sen_emb_class, params = best_parameters


    print ("\nEvaluating model for word embedding: {:s}({:s}), sentence embedding: {:s} \nHyperparameters {:s}\n"
           .format(word_emb_class.__name__, ', '.join(map(str, word_emb_params)), sen_emb_class.__name__, str(params)))
    params["n_jobs"] = n_jobs

    word_emb = word_emb_class(*word_emb_params)
    word_emb.build()
    sen_emb = sen_emb_class()
    sen_emb.build(word_emb)

    max_pca_dim = int(DATA_SIZE / float(FOLDS_COUNT) * (FOLDS_COUNT - 1)) # PCA dim can't be greater than samples count
    pca_lengths = [1,2,3] + \
                  map(int, np.linspace(5, min(max_pca_dim, sen_emb.word_vector_length), num=pca_count - 3))

    for pca_length in pca_lengths:
        print ("Calculating model for PCA with dimensions reduced to {0}..".format(pca_length))
        sen_emb = sen_emb_class(pca_length)

        validation_results, training_time, testing_time = \
            test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb, classifier_class,
                                  FOLDS_COUNT, False, True, **params)
        pca_accuracies.append(sum(validation_results) / FOLDS_COUNT)
        training_times.append(training_time)
        testing_times.append(testing_time)


    # last element will be time/accuracy without PCA
    print ("Calculating model with no PCA")
    pca_lengths.append(sen_emb.word_vector_length)
    sen_emb = sen_emb_class()

    validation_results, training_time, testing_time = \
        test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb, classifier_class,
                              FOLDS_COUNT, False, True, **params)
    pca_accuracies.append(sum(validation_results) / FOLDS_COUNT)
    training_times.append(training_time)
    testing_times.append(testing_time)

    highest_time = max([max(training_times), max(testing_times)])

    # normalize times
    for i in xrange(len(pca_lengths)):
        training_times[i] /= highest_time
        testing_times[i] /= highest_time

    with open(output_path, 'w') as output_file:
        for i, pca_length in enumerate(pca_lengths):
            output_file.write('{0};{1};{2};{3}\n'
                              .format(pca_length, pca_accuracies[i], training_times[i], testing_times[i]))

    return pca_lengths, pca_accuracies, training_times, testing_times


def plot_pca_accuracies(classifier_class, pca_lengths, pca_accuracies, training_times, testing_times):
    accuracy_legend = mpatches.Patch(color='b', label="Accuracy")
    training_time_legend = mpatches.Patch(color='r', label="Relative model training time")
    testing_time_legend = mpatches.Patch(color='g', label="Relative predicting time using model")
    no_pca_legend = mpatches.Patch(color='black', label="* = no PCA")

    plt.rcParams["figure.figsize"] = [11, 8]
    plt.legend(handles=[accuracy_legend, training_time_legend, testing_time_legend, no_pca_legend])
    lines = plt.plot(pca_lengths[:-1], pca_accuracies[:-1], 'b', pca_lengths[:-1], pca_accuracies[:-1], 'bo',
                     pca_lengths[:-1], training_times[:-1], 'r', pca_lengths[:-1], training_times[:-1], 'ro',
                     pca_lengths[:-1], testing_times[:-1], 'g', pca_lengths[:-1], testing_times[:-1], 'go')
    plt.setp(lines, linewidth=2, markersize=8)
    plt.scatter(pca_lengths[-1:], pca_accuracies[-1:], c='b', s=150, marker='*', edgecolors='black')
    plt.scatter(pca_lengths[-1:], training_times[-1:], c='r', s=150, marker='*', edgecolors='black')
    plt.scatter(pca_lengths[-1:], testing_times[-1:], c='g', s=150, marker='*', edgecolors='black')

    plt.title('How PCA dimension reduction affects model accuracy (using {0})?'.format(classifier_class.__name__))
    plt.xlabel('PCA dimensions')
    plt.ylabel('Accuracy vs execution time')
    save_current_plot('pca_{0}.svg'.format(classifier_class.__name__))
    plt.show()
    
    
def read_pca_results_file(file_path):
    pca_lengths = []
    pca_accuracies = []
    pca_training_times = []
    pca_testing_times = []
    for line in open(file_path, 'r'):
        pca_length, pca_result, pca_training_time, pca_testing_time = tuple(line.strip().split(";")[:4])
        pca_lengths.append(int(pca_length))
        pca_accuracies.append(float(pca_result))
        pca_training_times.append(float(pca_training_time))
        pca_testing_times.append(float(pca_testing_time))
    return pca_lengths, pca_accuracies, pca_training_times, pca_testing_times
    

def visualize_pca(classifier_class, n_jobs):
    pca_file_path = get_pca_results_path(DATA_FOLDER, classifier_class)
    use_new_pca = False

    if not (os.path.exists(pca_file_path) and os.path.isfile(pca_file_path)) or os.stat(pca_file_path).st_size == 0:
        print "PCA Results summary file does not exist or is empty. New search will be run."
        use_new_pca = True
    else:
        answer = raw_input("Do you wish to use existing PCA results summary file [y] or run new calculations? [n] ")
        if answer.lower() == 'y' or answer.lower == 'yes':
            print ("Using existing PCA summary file...")
        else:
            use_new_pca = True

    if use_new_pca:
        pca_count = 30 # may be changed
        pca_lengths, pca_accuracies, pca_training_times, pca_testing_times = \
            calculate_pca_accuracies(classifier_class, pca_count, n_jobs)
    else:
        pca_lengths, pca_accuracies, pca_training_times, pca_testing_times = \
            read_pca_results_file(pca_file_path)
    
    plot_pca_accuracies(classifier_class, pca_lengths, pca_accuracies, pca_training_times, pca_testing_times)


if __name__ == "__main__":
    classifier_class = choose_classifier()
    visualize_pca(classifier_class, n_jobs=-1)

