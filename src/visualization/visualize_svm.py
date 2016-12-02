from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.sentence_embeddings import sentence_embeddings
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.test_parameters import test_all_params_combinations, log_range, get_test_summary_path
from src.data import make_dataset
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import os
import ast

if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    folds_count = 5
    svmAlgorithm = SvmAlgorithm()
    word_embeddings = [Word2VecEmbedding(TextCorpora.get_corpus("brown"))]
    sentence_embeddings = [
        sentence_embeddings.ConcatenationEmbedding(),
        sentence_embeddings.SumEmbedding(),
        sentence_embeddings.TermCategoryVarianceEmbedding(),
        sentence_embeddings.ReverseTermFrequencyAverageEmbedding(),
        sentence_embeddings.TermFrequencyAverageEmbedding()
    ]
    c_range = log_range(0, 4)
    gamma_range = log_range(-3, 0)

    summary_file_path = get_test_summary_path(data_folder, svmAlgorithm)

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

        test_all_params_combinations(data_folder, svmAlgorithm, folds_count,
                                     word_embeddings=word_embeddings,
                                     sentence_embeddings=sentence_embeddings,
                                     params={"C": c_range, "gamma": gamma_range})

    summary = {}
    for line in open(summary_file_path, 'r'):
        embedding, params, result = tuple(line.split(";"))
        params_dict = ast.literal_eval(params)
        if embedding not in summary:
            summary[embedding] = [(params_dict, result)]
        else:
            summary[embedding].append((params_dict, result))

    for embedding, results, in summary.iteritems():
        fig = plt.figure()
        plt.title(embedding)
        plt.xlabel('C')
        plt.ylabel('gamma')
        ax = fig.gca(projection='3d')

        X, Y, Z = [], [], []

        for params, result in results:
            X.append(float(params['C']))
            Y.append(float(params['gamma']))
            Z.append(float(result))

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()




