from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.word2vec_embedding import *
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
from src.visualization.save_visualization import save_current_plot
from src.common import  DATA_SIZE, LABELS, CATEGORIES_COUNT, SENTENCES, CATEGORIES, SENTENCE_EMBEDDINGS


def visualize_sentence_embeddings(word_emb, sentence_embeddings):
    # take only 100 examples for each category for visualization
    examples_from_category = 100
    folds_count = int(DATA_SIZE / (examples_from_category * CATEGORIES_COUNT))

    folds_count = max(folds_count, 3)
    skf = StratifiedKFold(n_splits=folds_count)
    _, example_data_indices = next(skf.split(SENTENCES, LABELS))

    example_labels = LABELS[example_data_indices]
    example_sentences = SENTENCES[example_data_indices]

    word_emb.build()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Example of several sentence embeddings in action")
    gs = gridspec.GridSpec(1, len(sentence_embeddings))
    colors = ['r', 'g', 'b', 'yellow', 'magenta', 'cyan']
    legend_handles = []

    for i, sen_emb_class in enumerate(sentence_embeddings):
        sen_emb = sen_emb_class(3) # PCA to 3 dimensions
        print ("Building sentence embedding: " + type(sen_emb).__name__ + "...")
        sen_emb.build(word_emb, SENTENCES)

        example_sentences_vectors = [sen_emb[s] for s in example_sentences]

        ax = plt.subplot(gs[i], projection='3d')
        ax.set_title(type(sen_emb).__name__)

        colors_gen = itertools.cycle(colors)

        # plot dots representing sentences
        xs, ys, zs = [], [], []

        for j in xrange(CATEGORIES_COUNT):
            xs.append([])
            ys.append([])
            zs.append([])
            category_vectors = filter(lambda (k, s): example_labels[k] == j, enumerate(example_sentences_vectors))
            for k, sentence_vector in category_vectors:
                xs[j].append(sentence_vector[0])
                ys[j].append(sentence_vector[1])
                zs[j].append(sentence_vector[2])

            color = next(colors_gen)
            ax.scatter(xs[j], ys[j], zs[j], c=color, s=60, picker=True)

            if i == len(sentence_embeddings) - 1:
                legend_handles.append(mpatches.Patch(color=color, label=CATEGORIES[j]))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.legend(handles=legend_handles)
    plt.tight_layout()
    save_current_plot('sentence_embeddings.svg')
    plt.show()

if __name__ == "__main__":
    # use same word embedding for all classifiers - we don't want to load more than one embedding to RAM
    word_emb = Word2VecEmbedding('google/GoogleNews-vectors-negative300.bin', 300)
    visualize_sentence_embeddings(word_emb, SENTENCE_EMBEDDINGS)