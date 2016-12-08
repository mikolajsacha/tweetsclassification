import ast

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from sklearn.model_selection import StratifiedKFold
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    data_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    summary_file_path = get_grid_search_results_path(data_folder, SvmAlgorithm)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        exit(-1)

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to gest some results."
        exit(-1)

    max_result = 0.0
    best_parameters = []

    print "Found Grid Search results in " + summary_file_path.split("..")[-1]
    for line in open(summary_file_path, 'r'):
        embedding, params, result = tuple(line.split(";"))
        if result > max_result:
            max_result = result
            best_parameters = []
        if result >= max_result:
            params_dict = ast.literal_eval(params)
            best_parameters.append((embedding, params_dict))

    print "Model evaluation: {:4.2f}% for the following sets of parameters:\n".format(float(max_result))

    for embedding, params in best_parameters:
        print embedding, str(params)

    embedding, params = best_parameters[0]
    print "\nEvaluating model for embedding {:s} and parameters: {:s}...".format(embedding, str(params))

    word_emb_class, sen_emb_class = tuple(embedding.split(","))

    labels, sentences = make_dataset.read_dataset(data_path, data_info)

    print ("Building word and sentence embeddings...")
    word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
    word_emb.build(sentences)

    # for the sake of visualization we will use 3 dimensional sentence vectors
    # this gives model accuracy at about 60-80% but should be sufficient for a visualization
    ISentenceEmbedding.target_sentence_vector_length = 3
    sen_emb = eval(sen_emb_class)()
    sen_emb.build(word_emb, labels, sentences)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, labels, sentences)

    print ("Building model...")
    svmAlg = SvmAlgorithm(sen_emb, **params)
    svmAlg.fit(fb.features, fb.labels)

    print ("Drawing plot...")
    # take only 20 examples for each category for visualization
    examples_from_category = 20
    categories_count = len(data_info['Categories'])
    data_set_size = int(data_info['Size'])
    folds_count = int(data_set_size / (examples_from_category * categories_count))

    skf = StratifiedKFold(n_splits=folds_count)
    _, example_data_indices = next(skf.split(sentences, labels))

    example_labels = labels[example_data_indices]
    example_sentences = sentences[example_data_indices]
    example_sentences_vectors = [sen_emb[s] for s in example_sentences]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Data classification using PCA reducing sentence dimensions to 3")
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0], projection='3d')
    ax.set_title('Real labels of some training set samples')

    colors = ['r', 'g', 'b', 'yellow', 'magenta', 'cyan']
    colors_gen = itertools.cycle(colors)
    legend_handles = []

    # plot dots representing tested sentences
    xs, ys, zs = [], [], []
    underlying_sentences = []

    def on_dot_pick(event):
        set_num = int(event.artist.get_label()[-1])
        if len(underlying_sentences) > set_num:
            for i in (int(j) for j in list(event.ind)):
                print underlying_sentences[set_num][i]

    for i in xrange(categories_count):
        xs.append([])
        ys.append([])
        zs.append([])
        underlying_sentences.append([])
        category_vectors = filter(lambda (j, s): example_labels[j] == i, enumerate(example_sentences_vectors))
        for j, sentence_vector in category_vectors:
            xs[i].append(sentence_vector[0])
            ys[i].append(sentence_vector[1])
            zs[i].append(sentence_vector[2])
            underlying_sentences[i].append(example_sentences[j])

        color = next(colors_gen)
        ax.scatter(xs[i], ys[i], zs[i], c=color, s=60, picker=True)
        legend_handles.append(mpatches.Patch(color=color, label=data_info['Categories'][i]))

    fig.canvas.mpl_connect('pick_event', on_dot_pick)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plot hyperplane acquired by SVM
    X_MIN = min(s[0] for s in example_sentences_vectors)
    X_MAX = max(s[0] for s in example_sentences_vectors)
    Y_MIN = min(s[1] for s in example_sentences_vectors)
    Y_MAX = max(s[1] for s in example_sentences_vectors)
    Z_MIN = min(s[2] for s in example_sentences_vectors)
    Z_MAX = max(s[2] for s in example_sentences_vectors)

    MESHGRID_SIZE = 5

    xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, MESHGRID_SIZE),
                             np.linspace(Y_MIN, Y_MAX, MESHGRID_SIZE),
                             np.linspace(Z_MIN, Z_MAX, MESHGRID_SIZE))
    sens = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = svmAlg.clf.predict(sens)

    ax = plt.subplot(gs[1], projection='3d')
    ax.set_title('Visualization of hyperplane acquire by SVM')

    xs, ys, zs = [], [], []
    colors_gen = itertools.cycle(colors)
    for i in xrange(categories_count):
        xs.append([])
        ys.append([])
        zs.append([])
        category_vectors = filter(lambda (j, s): Z[j] == i, enumerate(sens))
        for j, sentence_vector in category_vectors:
            xs[i].append(sentence_vector[0])
            ys[i].append(sentence_vector[1])
            zs[i].append(sentence_vector[2])

        color = next(colors_gen)
        ax.scatter(xs[i], ys[i], zs[i], c=color, s=60)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    print ("Click on a dot to display a sentence it is representing")
    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.show()
