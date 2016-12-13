import ast

from matplotlib.colors import ListedColormap

from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from src.visualization.save_visualization import save_current_plot
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":
    data_folder = "dataset3_reduced"
    data_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    summary_file_path = get_grid_search_results_path(data_folder, RandomForestAlgorithm)

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

    labels, sentences = make_dataset.read_dataset(data_path, data_info)
    embedding, params = best_parameters[0]
    word_emb_class, sen_emb_class = tuple(embedding.split(","))

    print ("\nEvaluating model for embedding {:s} with params {:s}".format(embedding, str(params)))
    print ("Calculating model as a set of binary classifiers...")

    print ("Building word embedding...")
    word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
    word_emb.build(sentences)

    # for the sake of visualization we will use 2 dimensional sentence vectors
    ISentenceEmbedding.target_sentence_vector_length = 2
    sen_emb = eval(sen_emb_class)()

    print ("Building sentence embedding...")
    sen_emb.build(word_emb, labels, sentences)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, labels, sentences)

    print ("Building model...")
    rf = RandomForestAlgorithm(sen_emb, **params)
    rf.fit(fb.features, fb.labels)

    print ("Drawing plot...")

    xs, ys = [], []
    categories_count = len(data_info['Categories'])
    for i in xrange(categories_count):
        category_vectors = filter(lambda (k, s): fb.labels[k] == i, enumerate(fb.features))
        xs.append([vec[0] for _, vec in category_vectors])
        ys.append([vec[1] for _, vec in category_vectors])

    x_min, x_max = min(min(x) for x in xs), max(max(x) for x in xs)
    y_min, y_max = min(min(y) for y in ys), max(max(y) for y in ys)
    MESHGRID_SIZE = 10
    xx, yy = np.meshgrid(np.linspace(x_min - 0.1, x_max + 0.1, MESHGRID_SIZE),
                         np.linspace(y_min - 0.1, y_max + 0.1, MESHGRID_SIZE))

    plt.rcParams["figure.figsize"] = [11, 8]
    fig, ax = plt.subplots()
    colors = ['r', 'y', 'b', 'g', 'cyan', 'magenta']
    colors_gen = itertools.cycle(colors)

    estimator_alpha = 1.0 / len(rf.clf.estimators_)
    color_map = ListedColormap(list(itertools.islice(colors_gen, categories_count)), name='classifiers_color_map')

    for tree in rf.clf.estimators_:
        z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, alpha=estimator_alpha, cmap=color_map)

    def on_click(event):
        x, y = event.xdata, event.ydata
        print "Point {0}, {1}:".format(x, y)
        proba = rf.clf.predict_proba([[x, y]])[0]
        print "Classifier prediction: {0}\n" \
              .format(", ".join(
                     [data_info['Categories'][i] + ": {:4.2f}%".format(100 * p) for i, p in enumerate(proba)]))

    def on_pick(event):
        if event.mouseevent.button != 1:
            return  # only left-click
        thisline = event.artist
        xdata, ydata = thisline.get_data()
        x, y = xdata[event.ind[0]], ydata[event.ind[0]]
        fit_sen = [sentences[i] for i, _ in filter(lambda (i, s): s[0] == x and s[1] == y, enumerate(fb.features))]
        if fit_sen:
            print "\n".join(" ".join(s) for s in fit_sen)

    legend_handles = []
    colors_gen = itertools.cycle(colors)

    for i, category in enumerate(data_info['Categories']):
        color = next(colors_gen)
        ax.plot(xs[i], ys[i], 'o', picker=5, color=color)
        legend_handles.append(mpatches.Patch(color=color, label=category))

    fig.canvas.callbacks.connect('button_press_event', on_click)
    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.suptitle("Random Forest method with PCA reduction to 2 dimensions")
    plt.legend(handles=legend_handles)
    save_current_plot('random_forest_visualization.svg')
    print ""
    plt.show()
