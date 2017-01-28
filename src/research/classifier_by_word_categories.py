from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.common import SENTENCES, LABELS, CATEGORIES, CATEGORIES_COUNT
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from word_categories import get_words_categories_regressors


def show_cross_validation_result(folds_count, n_estimators):
    skf = StratifiedKFold(n_splits=folds_count)

    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))

    success_rates_in_folds = []
    print("Cross-validating model...")
    for fold, (train_index, test_index) in enumerate(skf.split(SENTENCES, LABELS)):
        training_labels = LABELS[train_index]
        test_labels = LABELS[test_index]
        training_sentences = SENTENCES[train_index]

        word_emb.build(training_sentences)

        word_classifiers = get_words_categories_regressors(training_labels, training_sentences, word_emb)

        words_vectors = []
        w_indices = {}
        visited_words = set()
        for sentence in SENTENCES:
            for word in sentence:
                if word not in visited_words:
                    visited_words.add(word)
                    if word in word_emb.model:
                        w_indices[word] = len(words_vectors)
                        words_vectors.append(word_emb[word])

        w_predictions = [cls.predict(words_vectors) for cls in word_classifiers]

        features = np.zeros((len(SENTENCES), 4 * CATEGORIES_COUNT), dtype=float)
        for i, sentence in enumerate(SENTENCES):
            for j, cls in enumerate(word_classifiers):
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

        training_features = features[train_index]
        test_features = features[test_index]

        cls = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        cls.fit(training_features, training_labels)

        predictions = cls.predict(test_features)
        successes = sum(1 for i, prediction in enumerate(predictions) if prediction == test_labels[i])

        success_rate = float(successes) / len(test_labels)
        success_rates_in_folds.append(success_rate)

    avg_success_rate = sum(success_rates_in_folds) / folds_count * 100
    print "Average success rate: {:4.4f}%".format(avg_success_rate)


if __name__ == "__main__":
    """ Let's use classifiers for word categories to predict sentences categories"""
    n_estimators = 20
    # show_cross_validation_result(5, n_estimators)

    print "Training model on whole data-set for visualization..."
    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    word_emb.build(SENTENCES)

    word_classifiers = get_words_categories_regressors(LABELS, SENTENCES, word_emb)

    words_vectors = []
    w_indices = {}
    visited_words = set()
    for sentence in SENTENCES:
        for word in sentence:
            if word not in visited_words:
                visited_words.add(word)
                if word in word_emb.model:
                    w_indices[word] = len(words_vectors)
                    words_vectors.append(word_emb[word])

    w_predictions = [cls.predict(words_vectors) for cls in word_classifiers]

    features = np.zeros((len(SENTENCES), 4 * CATEGORIES_COUNT), dtype=float)
    for i, sentence in enumerate(SENTENCES):
        for j, cls in enumerate(word_classifiers):
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
