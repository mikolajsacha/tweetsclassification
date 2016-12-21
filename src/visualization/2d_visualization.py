import ast

from matplotlib import gridspec
from matplotlib.colors import ListedColormap

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from src.visualization.save_visualization import save_current_plot
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    data_folder = "dataset3"
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    categories_count = len(data_info['Categories'])
    data_path = make_dataset.get_processed_data_path(data_folder)
    possible_classifiers = [SvmAlgorithm, RandomForestAlgorithm, NeuralNetworkAlgorithm]
    labels, sentences = make_dataset.read_dataset(data_path, data_info)

    print("Choose classifiers to compare by typing a list of their indices (e.g: '0,2'):")
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(possible_classifiers))

    numbers = []
    while True:
        try:
            str_numbers = raw_input().replace(" ", "").split(',')
            for str_number in str_numbers:
                number = int(str_number)
                if len(possible_classifiers) > number >= 0:
                    numbers.append(number)
                else:
                    raise ValueError()
            break
        except ValueError:
            print "Please insert a correct list of numbers"

    classifiers = [possible_classifiers[i] for i in numbers]
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
    color_map = ListedColormap(list(itertools.islice(colors_gen, categories_count)), name='classifiers_color_map')

    for category in data_info['Categories']:
        color = next(colors_gen)
        legend_handles.append(mpatches.Patch(color=color, label=category))
        plt.legend(handles=legend_handles)

    for classifier_index, Classifier in enumerate(classifiers):
        summary_file_path = get_grid_search_results_path(data_folder, Classifier)

        if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
            print "Grid Search summary file does not exist. Please run grid_search.py at first."
            exit(-1)

        if os.stat(summary_file_path).st_size == 0:
            print "Grid Search summary file is empty. Please run grid_search.py to gest some results."
            exit(-1)

        max_result = 0.0
        best_parameters = []

        print("Found Grid Search results in " + summary_file_path.split("..")[-1])
        for line in open(summary_file_path, 'r'):
            embedding, params, result = tuple(line.split(";"))
            if result > max_result:
                max_result = result
                best_parameters = []
            if result >= max_result:
                params_dict = ast.literal_eval(params)
                best_parameters.append((embedding, params_dict))

        print("Model evaluation: {:4.2f}% for the following embeddings and parameters:\n".format(float(max_result)))
        for embedding, params in best_parameters:
            print embedding, params

        embedding, params = best_parameters[0]
        word_emb_class, sen_emb_class = tuple(embedding.split(","))

        print ("\nEvaluating model for embedding {:s} with params {:s}".format(embedding, str(params)))
        print ("Calculating model as a set of binary classifiers...")

        print ("Building word embedding...")
        word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
        word_emb.build(sentences)

        # for the sake of visualization we will use 2 dimensional sentence vectors
        sen_emb = eval(sen_emb_class)(2)

        print ("Building sentence embedding...")
        sen_emb.build(word_emb, labels, sentences)

        print ("Building features...")
        fb = build_features.FeatureBuilder()
        fb.build(sen_emb, labels, sentences)
        classifiers_features.append(fb.features)

        print ("Building model...")
        params['probability'] = True
        classifier = Classifier(sen_emb, **params)
        classifier.fit(fb.features, fb.labels)
        trained_classifiers.append(classifier)

        print ("Drawing plot...")

        xs, ys = [], []
        for i in xrange(categories_count):
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

        for i, category in enumerate(data_info['Categories']):
            ax.scatter(xs[i], ys[i], color=next(colors_gen), picker=True, edgecolors='black', s=30)

    def on_click(event):
        x, y = event.xdata, event.ydata
        print "Point {:5.4f}, {:5.4f}:".format(x, y)
        for classifier in trained_classifiers:
            proba = classifier.clf.predict_proba([[x, y]])[0]
            print "{0} prediction: {1}" \
                  .format(type(classifier).__name__, ", ".join(
                         [data_info['Categories'][i] + ": {:2.0f}%".format(100 * p) for i, p in enumerate(proba)]))
        print "\n"


    # noinspection PyProtectedMember
    def on_pick(event):
        try:
            if event.mouseevent.button != 1:
                return  # only left-click
            for ind in event.ind:
                print " ".join(sentences[ind])
        except Exception as e: print e

    fig.canvas.callbacks.connect('button_press_event', on_click)
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.suptitle("Visualization of chosen classification algorithms with number of dimensions reduced to 2")
    plt.tight_layout()
    save_current_plot('2d_visualization.svg')
    print ""
    plt.show()
