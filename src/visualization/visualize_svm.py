import math

import itertools

from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.test_parameters import test_all_params_combinations, log_range, get_test_summary_path
from src.data import make_dataset
import mpl_toolkits.mplot3d as mplot3d
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import os
import ast
import numpy as np

if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    folds_count = 5
    training_set_size = 0.80
    word_embeddings = [Word2VecEmbedding(TextCorpora.get_corpus("brown"))]
    sentence_embeddings = [
        sentence_embeddings.SumEmbedding(),
        sentence_embeddings.TermCategoryVarianceEmbedding(),
        sentence_embeddings.TermFrequencyAverageEmbedding()
    ]
    c_range = log_range(0, 6)
    gamma_range = log_range(-3, 2)

    summary_file_path = get_test_summary_path(data_folder, SvmAlgorithm)

    generate_params = not os.path.isfile(summary_file_path)

    if not generate_params:
        command = raw_input("Use existing file [y] or test all parameters again (might take a couple of minutes) [n]? ")
        if command.lower() == "n":
            generate_params = True
    if generate_params:
        input_file_path = make_dataset.get_external_data_path(data_folder)
        output_file_path = make_dataset.get_processed_data_path(data_folder)

        if not os.path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)
            exit(-1)
        else:
            make_dataset.make_dataset(input_file_path, output_file_path)

        test_all_params_combinations(data_folder, SvmAlgorithm, folds_count, training_set_size,
                                     word_embeddings=word_embeddings,
                                     sentence_embeddings=sentence_embeddings,
                                     params={"C": c_range, "gamma": gamma_range})

    summary = {}
    for line in open(summary_file_path, 'r'):
        embedding, params, result = tuple(line.split(";"))
        params_dict = ast.literal_eval(params)
        if embedding not in summary:
            summary[embedding] = {(math.log(params_dict['C'], 10), math.log(params_dict['gamma'], 10)): result}
        else:
            summary[embedding][(math.log(params_dict['C'], 10), math.log(params_dict['gamma'], 10))] = result

    fig = plt.figure()

    ax = fig.gca(projection='3d')

    ax.set_xlabel('log(C)')
    ax.set_ylabel('log(gamma)')
    fig.suptitle("Model selection results acquired by cross-validation for different embeddings and parameters")
    colors = itertools.cycle(["r", "g", "b", "y", "cyan", "magenta"])
    legend_handles = []

    for embedding, results, in summary.iteritems():
        x, y = map(lambda e: sorted(set(e)), zip(*results.iterkeys()))
        x, y = np.meshgrid(x, y)
        z = np.empty(x.shape, dtype=float)

        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                z[i][j] = results[(x[i][j], y[i][j])]

        color = next(colors)
        legend_handles.append(mpatches.Patch(color=color, label=embedding))
        ax.plot_surface(x, y, z, color=color, linewidth=0, rstride=1, cstride=1)

    plt.legend(handles=legend_handles)
    plt.show()
