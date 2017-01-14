import ast
from matplotlib import gridspec
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.visualization.save_visualization import save_current_plot
from src.configuration import DATA_FOLDER

if __name__ == "__main__":
    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, SvmAlgorithm)

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

    sen_emb = eval(sen_emb_class)(3)

    categories_count = len(data_info['Categories'])

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Data classification using PCA reducing sentence dimensions to 3")
    gs = gridspec.GridSpec(1, categories_count)

    colors = ['r', 'g', 'b', 'yellow', 'magenta', 'cyan']
    color_gen = itertools.cycle(colors)
    legend_handles = []

    MESHGRID_SIZE = 10


    def map_label_to_binary_for(index):
        def map_label_to_binary(label):
            return 1 if label == index else 0

        return map_label_to_binary


    xs = [[]] * categories_count
    ys = [[]] * categories_count
    zs = [[]] * categories_count

    svm_models = []

    for i, category in enumerate(data_info['Categories']):
        print ("Evaluating {0}/{1} binary classifer...".format(i + 1, categories_count))

        map_fun = map_label_to_binary_for(i)
        mapped_labels = np.array(map(map_fun, labels))

        print ("Building sentence embedding...")
        sen_emb.build(word_emb, mapped_labels, sentences)

        print ("Building features...")
        fb = build_features.FeatureBuilder()
        fb.build(sen_emb, mapped_labels, sentences)

        print ("Building model...")
        svm = SvmAlgorithm(sen_emb, **params)
        svm.fit(fb.features, fb.labels)
        svm_models.append(svm)

        category_vectors = filter(lambda (k, s): fb.labels[k] == 1, enumerate(fb.features))
        for k, sentence_vector in category_vectors:
            xs[i].append(sentence_vector[0])
            ys[i].append(sentence_vector[1])
            zs[i].append(sentence_vector[2])

    print ("Drawing plot...")

    def flatten(l):
        return (item for sublist in l for item in sublist)

    X_MIN = min(flatten(xs))
    X_MAX = max(flatten(xs))
    Y_MIN = min(flatten(ys))
    Y_MAX = max(flatten(ys))
    Z_MIN = min(flatten(zs))
    Z_MAX = max(flatten(zs))

    # plot dots representing sentences
    for i, category in enumerate(data_info['Categories']):
        ax = plt.subplot(gs[i], projection='3d')
        ax.set_title(category)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_zlim(Z_MIN, Z_MAX)

        color = next(color_gen)
        legend_handles.append(mpatches.Patch(color=color, label=category))

        ax.scatter(xs[i], ys[i], zs[i], c=color, s=40)

        # plot hyperplane acquired by SVM
        xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, MESHGRID_SIZE),
                                 np.linspace(Y_MIN, Y_MAX, MESHGRID_SIZE),
                                 np.linspace(Z_MIN, Z_MAX, MESHGRID_SIZE))
        Z = svm_models[i].clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        Z = Z.reshape(xx.shape)

        verts, faces = measure.marching_cubes(Z, (Z.min() + Z.max())/2.0)
        verts = verts * [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / MESHGRID_SIZE
        verts = verts + [X_MIN, Y_MIN, Z_MIN]
        mesh = Poly3DCollection(verts[faces], facecolor=color, edgecolor=color, alpha=0.5)

        ax.add_collection3d(mesh)

    plt.legend(handles=legend_handles)
    plt.tight_layout()
    save_current_plot('svm_visualization.svg')
    plt.show()
