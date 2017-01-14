from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.word2vec_embedding import *
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
from src.data import make_dataset
from src.features.sentence_embeddings import sentence_embeddings
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.visualization.save_visualization import save_current_plot
from src.configuration import DATA_FOLDER

if __name__ == "__main__":
    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))

    labels, sentences = make_dataset.read_dataset(data_path, data_info)

    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    sentence_embeddings = [
        sentence_embeddings.ConcatenationEmbedding(3),
        sentence_embeddings.SumEmbedding(3),
        sentence_embeddings.TermFrequencyAverageEmbedding(3)
    ]

    # take only 100 examples for each category for visualization
    examples_from_category = 100
    categories_count = len(data_info['Categories'])
    data_set_size = int(data_info['Size'])
    folds_count = int(data_set_size / (examples_from_category * categories_count))

    folds_count = max(folds_count, 3)
    skf = StratifiedKFold(n_splits=folds_count)
    _, example_data_indices = next(skf.split(sentences, labels))

    example_labels = labels[example_data_indices]
    example_sentences = sentences[example_data_indices]

    print ("Building word embedding...")
    word_emb.build(sentences)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Example of several sentence embeddings in action")
    gs = gridspec.GridSpec(1, len(sentence_embeddings))
    colors = ['r', 'g', 'b', 'yellow', 'magenta', 'cyan']
    legend_handles = []

    for i, sen_emb in enumerate(sentence_embeddings):
        print ("Building sentence embedding: " + type(sen_emb).__name__ + "...")
        sen_emb.build(word_emb, labels, sentences)

        example_sentences_vectors = [sen_emb[s] for s in example_sentences]

        ax = plt.subplot(gs[i], projection='3d')
        ax.set_title(type(sen_emb).__name__)

        colors_gen = itertools.cycle(colors)

        # plot dots representing sentences
        xs, ys, zs = [], [], []

        for j in xrange(categories_count):
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
                legend_handles.append(mpatches.Patch(color=color, label=data_info['Categories'][j]))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.legend(handles=legend_handles)
    plt.tight_layout()
    save_current_plot('sentence_embeddings.svg')
    plt.show()
