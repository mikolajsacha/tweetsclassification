from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.common import SENTENCES, LABELS, CATEGORIES, CATEGORIES_COUNT
from src.data import make_dataset
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.research.grid_search_sentence_classifier import get_best_own_classifier_from_grid_search_results
from src.research.grid_search_word_classifier import get_best_words_categories_regressors
from word_categories import get_words_categories_regressors, get_words_scores

if __name__ == "__main__":
    """ Let's use classifiers for word categories to predict sentences categories"""
    n_estimators = 20

    print "Training model on whole data-set for visualization..."
    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    word_emb.build(SENTENCES)

    word_scores = get_words_scores(LABELS, SENTENCES)
    word_vectors = [word_emb[word] for word in word_scores.iterkeys()]

    print("Building word regressors...")
    word_regressors = get_best_words_categories_regressors(word_scores, word_vectors, word_emb, verbose=True)

    w_predictions = [cls.predict(word_vectors) for cls in word_regressors]
    w_indices = {}
    for i, word in enumerate(word_scores.iterkeys()):
        w_indices[word] = i

    features = np.zeros((len(SENTENCES), 4 * CATEGORIES_COUNT), dtype=float)
    for i, sentence in enumerate(SENTENCES):
        for j, cls in enumerate(word_regressors):
            predictions = [w_predictions[j][w_indices[word]] for word in sentence if word in word_emb.model]
            if not predictions:
                continue
            pred_len = len(predictions)
            pred_avg = sum(predictions) / pred_len
            std_deviation = (sum(x ** 2 for x in predictions) / pred_len - (pred_avg / pred_len) ** 2) ** 0.5
            features[i][4 * j] = pred_avg
            features[i][4 * j + 1] = min(predictions)
            features[i][4 * j + 2] = max(predictions)
            features[i][4 * j + 3] = std_deviation

    # 2d visualization similar to that from visualization/2d_visualization.py
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    print ("Drawing plot...")

    xs, ys = [], []
    for i in xrange(CATEGORIES_COUNT):
        category_vectors = [vec for j, vec in enumerate(pca_features) if LABELS[j] == i]
        xs.append([vec[0] for vec in category_vectors])
        ys.append([vec[1] for vec in category_vectors])

    x_min, x_max = min(min(x) - 10 for x in xs), max(max(x) + 10 for x in xs)
    y_min, y_max = min(min(y) - 10 for y in ys), max(max(y) + 10 for y in ys)
    MESHGRID_SIZE = 300
    xx, yy = np.meshgrid(np.linspace(x_min - 0.1, x_max + 0.1, MESHGRID_SIZE),
                         np.linspace(y_min - 0.1, y_max + 0.1, MESHGRID_SIZE))

    colors = ['r', 'y', 'b', 'g', 'cyan', 'magenta']
    colors_gen = itertools.cycle(colors)
    color_map = ListedColormap(list(itertools.islice(colors_gen, CATEGORIES_COUNT)), name='classifiers_color_map')
    legend_handles = []

    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("")

    ax = plt.subplot()

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    clf.fit(pca_features, LABELS)
    estimator_alpha = 1.0 / len(clf.estimators_)

    for tree in clf.estimators_:
        z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        ax.contourf(xx, yy, z, alpha=estimator_alpha, cmap=color_map)

    colors_gen = itertools.cycle(colors)
    for i, category in enumerate(CATEGORIES):
        ax.scatter(xs[i], ys[i], color=next(colors_gen), picker=2, edgecolors='black', s=30)

    colors_gen = itertools.cycle(colors)
    for category in CATEGORIES:
        color = next(colors_gen)
        legend_handles.append(mpatches.Patch(color=color, label=category))


    def on_pick(event):
        ind = event.ind
        xy = event.artist.get_offsets()
        for x, y in xy[ind]:
            matching_indices = [k for k, f in enumerate(pca_features) if f[0] == x and f[1] == y]
            for index in matching_indices:
                print " ".join(SENTENCES[index])


    def on_click(event):
        x, y = event.xdata, event.ydata
        prediction = clf.predict_proba([[x, y]])[0]
        print ', '.join(CATEGORIES[i] + ": {:4.2f}%".format(pred*100) for i, pred in enumerate(prediction))


    fig.canvas.mpl_connect('pick_event', on_pick)
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.suptitle("Visualization of trained model with PCA to 2 dimensions")
    plt.legend(handles=legend_handles)
    plt.show()

    print ("\n\nInteractive test.")
    clf = get_best_own_classifier_from_grid_search_results(features, LABELS)
    if clf is None:
        exit(-1)

    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = make_dataset.string_to_words_list(command)
        sentence_features = np.zeros((4 * CATEGORIES_COUNT), dtype=float)
        skip = False
        for j, cls in enumerate(word_regressors):
            predictions = [cls.predict([word_emb[word]])[0] for word in sentence if word in word_emb.model]
            if not predictions:
                print "No word from sentence is in word embedding"
                skip = True
                break
            pred_len = len(predictions)
            pred_avg = sum(predictions) / pred_len
            std_deviation = (sum(x ** 2 for x in predictions) / pred_len - (pred_avg / pred_len) ** 2) ** 0.5
            sentence_features[4 * j] = pred_avg
            sentence_features[4 * j + 1] = min(predictions)
            sentence_features[4 * j + 2] = max(predictions)
            sentence_features[4 * j + 3] = std_deviation

        if not skip:
            print ', '.join("{:s}: {:4.2f}%".format(CATEGORIES[k], 100.0*prob)
                             for k, prob in enumerate(clf.predict_proba([sentence_features])[0]))
