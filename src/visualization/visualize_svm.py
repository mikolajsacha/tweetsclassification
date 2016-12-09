import ast
from matplotlib import gridspec
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from sklearn.model_selection import StratifiedKFold
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    # for the sake of visualization we will use 3 dimensional sentence vectors
    # this gives model accuracy at about 60-80% but should be sufficient for a visualization
    ISentenceEmbedding.target_sentence_vector_length = 3
    sen_emb = eval(sen_emb_class)()

    # take only 50 examples for each category for visualization
    examples_from_category = 50
    categories_count = len(data_info['Categories'])
    data_set_size = int(data_info['Size'])
    folds_count = int(data_set_size / (examples_from_category * categories_count))

    skf = StratifiedKFold(n_splits=folds_count)
    _, example_data_indices = next(skf.split(sentences, labels))

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

    xs = np.empty((categories_count, examples_from_category), dtype=float)
    ys = np.empty((categories_count, examples_from_category), dtype=float)
    zs = np.empty((categories_count, examples_from_category), dtype=float)

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
        j = 0
        for k, sentence_vector in category_vectors:
            if j >= examples_from_category:
                break
            xs[i][j] = sentence_vector[0]
            ys[i][j] = sentence_vector[1]
            zs[i][j] = sentence_vector[2]
            j += 1

    print ("Drawing plot...")

    X_MIN = min(xs.ravel())
    X_MAX = max(xs.ravel())
    Y_MIN = min(ys.ravel())
    Y_MAX = max(ys.ravel())
    Z_MIN = min(zs.ravel())
    Z_MAX = max(zs.ravel())

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

        ax.scatter(xs[i], ys[i], zs[i], c=color, s=70)
        other_indices = np.concatenate((np.arange(i, dtype=int), np.arange(i+1, len(xs), dtype=int)))

        other_xs = xs[other_indices].ravel()
        other_ys = ys[other_indices].ravel()
        other_zs = ys[other_indices].ravel()
        ax.scatter(other_xs, other_ys, other_zs, c='gray', s=20)

        # plot hyperplane acquired by SVM
        xx, yy, zz = np.meshgrid(np.linspace(X_MIN, X_MAX, MESHGRID_SIZE),
                                 np.linspace(Y_MIN, Y_MAX, MESHGRID_SIZE),
                                 np.linspace(Z_MIN, Z_MAX, MESHGRID_SIZE))
        Z = svm_models[i].clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
        Z = Z.reshape(xx.shape)

        verts, faces = measure.marching_cubes(Z, 0)
        verts = verts * [X_MAX - X_MIN, Y_MAX - Y_MIN, Z_MAX - Z_MIN] / MESHGRID_SIZE
        verts = verts + [X_MIN, Y_MIN, Z_MIN]
        mesh = Poly3DCollection(verts[faces], facecolor=color, edgecolor=color, alpha=0.5)

        ax.add_collection3d(mesh)

    plt.legend(handles=legend_handles)
    plt.tight_layout()
    plt.show()
