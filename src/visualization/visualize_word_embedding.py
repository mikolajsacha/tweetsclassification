import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
from sklearn.decomposition import PCA
import numpy as np

from src.data.dataset import get_unique_words
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.visualization.save_visualization import save_current_plot
from src.common import SENTENCES, CATEGORIES_COUNT, LABELS, CATEGORIES, PROCESSED_DATA_PATH


def mix_colors(weighted_color_list, colors):
    total_weight = float(sum(map(lambda x: x**3, weighted_color_list)))
    r, g, b = 0.0, 0.0, 0.0
    for i, weight in enumerate(weighted_color_list):
        col = colors[i]
        r += col[0] * (weight ** 3)
        g += col[1] * (weight ** 3)
        b += col[2] * (weight ** 3)
    return r / total_weight, g / total_weight, b / total_weight


def visualize_word_embedding(word_emb):
    print ("Training PCA on words from dataset...")


    dataset_words = get_unique_words(PROCESSED_DATA_PATH)
    dataset_words_with_emb = [word_emb[w] for w in dataset_words if word_emb[w] is not None]
    print("{:d}/{:d} ({:4.2f}% words from dataset exist in embedding"
          .format(len(dataset_words_with_emb), len(dataset_words),
                  len(dataset_words_with_emb) / float(len(dataset_words)) * 100))

    pca = PCA(n_components=3)
    pca.fit(dataset_words_with_emb)

    # take all the words from dataset and count their occurrences in categories
    # take only words which occur at least in 3 different tweets and are longer than 2 letters
    uniq_sens = (set(sen) for sen in SENTENCES)  # remove duplicate words from tweets
    words_with_tweets_counts = {}
    for i, sen in enumerate(uniq_sens):
        sen_category = LABELS[i]
        for word in filter(lambda x: len(x) > 2, sen):
            if word not in words_with_tweets_counts:
                words_with_tweets_counts[word] = [0] * CATEGORIES_COUNT
            words_with_tweets_counts[word][sen_category] += 1

    words_with_tweets_counts = filter(lambda (word, counters):
                                      sum(counters) >= 3 and word_emb[word] is not None,
                                      words_with_tweets_counts.iteritems())
    trimmed_words = []
    words_from_category = 75  # leave only 75 words from each "category"
    categories_counters = [0] * CATEGORIES_COUNT
    for word, counters in words_with_tweets_counts:
        word_category = counters.index(max(counters))
        if categories_counters[word_category] >= words_from_category:
            continue
        categories_counters[word_category] += 1
        trimmed_words.append((word, counters))

    words = list(map(lambda x: x[0], trimmed_words))

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # colors in RGB [0,1]
    words_colors = list(map(lambda x: mix_colors(x[1], colors), trimmed_words))

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Example of word embedding in action")
    legend_handles = []

    ax = plt.subplot(projection='3d')

    # plot dots representing words
    xs, ys, zs = [], [], []
    for word in words:
        word_vector = pca.transform([word_emb[word]])[0]
        xs.append(word_vector[0])
        ys.append(word_vector[1])
        zs.append(word_vector[2])

    ax.scatter(xs,ys, zs, c=words_colors, s=60, picker=True)

    for i in xrange(CATEGORIES_COUNT):
        legend_handles.append(mpatches.Patch(color=colors[i], label=CATEGORIES[i]))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    curr_tooltip = None

    def on_pick(event):
        try:
            if event.mouseevent.button != 1:
                return  # only left-click
            print ', '.join(words[ind] for ind in event.ind)
        except Exception as e:
            print "Exception on pick event: " + e

    fig.canvas.mpl_connect('pick_event', on_pick)

    plt.legend(handles=legend_handles)
    plt.tight_layout()
    save_current_plot('word_embedding.svg')
    plt.show()


if __name__ == "__main__":
    print ("Building word embedding...")
    word_emb = Word2VecEmbedding('google/GoogleNews-vectors-negative300.bin', 300)
    word_emb.build()

    visualize_word_embedding(word_emb)
