import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
from src.data import make_dataset
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.visualization.save_visualization import save_current_plot
from src.configuration import DATA_FOLDER


def mix_colors(weighted_color_list, colors):
    total_weight = float(sum(map(lambda x: x**3, weighted_color_list)))
    r, g, b = 0.0, 0.0, 0.0
    for i, weight in enumerate(weighted_color_list):
        col = colors[i]
        r += col[0] * (weight ** 3)
        g += col[1] * (weight ** 3)
        b += col[2] * (weight ** 3)
    return r / total_weight, g / total_weight, b / total_weight

if __name__ == "__main__":
    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))
    categories_count = len(data_info['Categories'])
    data_set_size = int(data_info['Size'])

    labels, sentences = make_dataset.read_dataset(data_path, data_info)

    print ("Building word embedding...")
    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"), 3)
    word_emb.build(sentences)

    print ("Drawing plot...")

    # take all the words from dataset and count their occurences in categories
    # take only words which occur at least in 3 different tweets and are longer than 2 letters
    uniq_sens = (set(sen) for sen in sentences)  # remove duplicate words from tweets
    words_with_tweets_counts = {}
    for i, sen in enumerate(uniq_sens):
        sen_category = labels[i]
        for word in filter(lambda x: len(x) > 2, sen):
            if word not in words_with_tweets_counts:
                words_with_tweets_counts[word] = [0] * categories_count
            words_with_tweets_counts[word][sen_category] += 1

    words_with_tweets_counts = filter(lambda (word, counters): sum(counters) >= 3, words_with_tweets_counts.iteritems())
    trimmed_words = []
    words_from_category = 75  # leave only 75 words from each "category"
    categories_counters = [0] * categories_count
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
        word_vector = word_emb[word]
        xs.append(word_vector[0])
        ys.append(word_vector[1])
        zs.append(word_vector[2])

    ax.scatter(xs, ys, zs, c=words_colors, s=60, picker=True)

    for i in xrange(categories_count):
        legend_handles.append(mpatches.Patch(color=colors[i], label=data_info['Categories'][i]))

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
