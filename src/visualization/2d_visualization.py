import ast

import multiprocessing
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.features.word_embeddings.keras_word_embedding import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_best_from_grid_search_results
from src.visualization.save_visualization import save_current_plot
from src.common import CATEGORIES_COUNT, CATEGORIES, SENTENCES, choose_multiple_classifiers, LABELS
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    classifiers = choose_multiple_classifiers();
    trained_classifiers = []
    classifiers_features = []
    subplots = []

    fig = plt.figure(figsize=(8 * len(classifiers), 8))
    fig.suptitle("Example of several sentence embeddings in action")
    gs = gridspec.GridSpec(1, len(classifiers))

    plt.rcParams["figure.figsize"] = [11, 8]
    colors = ['r', 'y', 'b', 'g', 'cyan', 'magenta']

    legend_handles = []
    colors_gen = itertools.cycle(colors)
    color_map = ListedColormap(list(itertools.islice(colors_gen, CATEGORIES_COUNT)), name='classifiers_color_map')

    for category in CATEGORIES:
        color = next(colors_gen)
        legend_handles.append(mpatches.Patch(color=color, label=category))

    for classifier_index, Classifier in enumerate(classifiers):
        best_parameters = get_best_from_grid_search_results(Classifier)
        if best_parameters is None:
            exit(-1)
        embedding, params = best_parameters
        word_emb_class, sen_emb_class = tuple(embedding.split(","))

        print ("\nEvaluating model for embedding {:s} with params {:s}".format(embedding, str(params)))
        params["n_jobs"] = multiprocessing.cpu_count()

        print ("Building word embedding...")
        word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
        word_emb.build(SENTENCES)

        # for the sake of visualization we will use 2 dimensional sentence vectors
        sen_emb = eval(sen_emb_class)(2)

        print ("Building sentence embedding...")
        sen_emb.build(word_emb, LABELS, SENTENCES)

        print ("Building features...")
        fb = build_features.FeatureBuilder()
        fb.build(sen_emb, LABELS, SENTENCES)
        classifiers_features.append(fb.features)

        print ("Building model...")
        params['probability'] = True
        classifier = Classifier(sen_emb, **params)
        classifier.fit(fb.features, fb.labels)
        trained_classifiers.append(classifier)

        print ("Drawing plot...")

        xs, ys = [], []
        for i in xrange(CATEGORIES_COUNT):
            category_vectors = filter(lambda (k, s): fb.labels[k] == i, enumerate(fb.features))
            xs.append([vec[0] for _, vec in category_vectors])
            ys.append([vec[1] for _, vec in category_vectors])

        x_min, x_max = min(min(x) for x in xs), max(max(x) for x in xs)
        y_min, y_max = min(min(y) for y in ys), max(max(y) for y in ys)
        MESHGRID_SIZE = 300
        xx, yy = np.meshgrid(np.linspace(x_min - 0.1, x_max + 0.1, MESHGRID_SIZE),
                             np.linspace(y_min - 0.1, y_max + 0.1, MESHGRID_SIZE))

        colors_gen = itertools.cycle(colors)

        ax = fig.add_subplot(gs[classifier_index])
        ax.text(.5, .92, Classifier.__name__, horizontalalignment='center', transform=ax.transAxes)
        subplots.append(ax)
        classifier.visualize_2d(xx, yy, ax, color_map)

        colors_gen = itertools.cycle(colors)

        for i, category in enumerate(CATEGORIES):
            ax.scatter(xs[i], ys[i], color=next(colors_gen), picker=True, edgecolors='black', s=30)

    def on_click(event):
        x, y = event.xdata, event.ydata
        print "Point {:5.4f}, {:5.4f}:".format(x, y)
        for classifier in trained_classifiers:
            proba = classifier.clf.predict_proba([[x, y]])[0]
            print "{0} prediction: {1}" \
                  .format(type(classifier).__name__, ", ".join(
                         [CATEGORIES[i] + ": {:2.0f}%".format(100 * p) for i, p in enumerate(proba)]))
        print "\n"


    fig.canvas.callbacks.connect('button_press_event', on_click)

    plt.suptitle("Visualization of chosen classification algorithms with number of dimensions reduced to 2")
    plt.tight_layout()
    plt.legend(handles=legend_handles)
    save_current_plot('2d_visualization.svg')
    print ""
    plt.show()
