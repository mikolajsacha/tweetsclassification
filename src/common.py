from src.data import make_dataset
from src.features.sentence_embeddings import sentence_embeddings
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.features.word_embeddings.keras_word_embedding import KerasWordEmbedding
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.nearest_neighbors_algorithm import NearestNeighborsAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def log_range(min_ten_power, max_ten_power, base=10):
    return (base ** i for i in xrange(min_ten_power, max_ten_power))


DATA_FOLDER = "gathered_dataset"
FOLDS_COUNT = 5
TRAINING_SET_SIZE = 0.80

CLASSIFIERS_PARAMS = [(SvmAlgorithm, {"C": list(log_range(0, 6)), "gamma": list(log_range(-3, 2))}),
                      (RandomForestAlgorithm, {"criterion": ["gini", "entropy"],
                                               "min_samples_split": [2, 10],
                                               "min_samples_leaf": [1, 10],
                                               "max_features": [None, "sqrt"]}),
                      (NeuralNetworkAlgorithm, {"alpha": list(log_range(-5, -2)),
                                                "learning_rate": ["constant", "adaptive"],
                                                "activation": ["identity", "logistic", "tanh", "relu"],
                                                "hidden_layer_sizes": [(100,), (100, 50)]}),
                      (NearestNeighborsAlgorithm, {'n_neighbors': [1, 2, 3, 4, 7, 15, 30], 'p': [1, 2, 3],
                                                   'weights': ['uniform', 'distance']})
                      ]

CLASSIFIERS = [c[0] for c in CLASSIFIERS_PARAMS]

WORD_EMBEDDINGS = [
    Word2VecEmbedding(TextCorpora.get_corpus("brown")),
    KerasWordEmbedding(TextCorpora.get_corpus("brown"))
]
SENTENCE_EMBEDDINGS = [
    sentence_embeddings.ConcatenationEmbedding(),
    sentence_embeddings.SumEmbedding(),
    sentence_embeddings.TermFrequencyAverageEmbedding()
]

EXTERNAL_DATA_PATH = make_dataset.get_external_data_path(DATA_FOLDER)
PROCESSED_DATA_PATH = make_dataset.get_processed_data_path(DATA_FOLDER)
DATA_INFO = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))
CATEGORIES = DATA_INFO['Categories']
DATA_SIZE = DATA_INFO['Size']
CATEGORIES_COUNT = len(CATEGORIES)
LABELS, SENTENCES = make_dataset.read_dataset(EXTERNAL_DATA_PATH, DATA_INFO)


def choose_multiple_classifiers():
    print("Choose classifiers by typing a list of their indices (e.g: '0,2'):")
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(CLASSIFIERS))
    numbers = []
    while True:
        try:
            str_numbers = raw_input().replace(" ", "").split(',')
            for str_number in str_numbers:
                number = int(str_number)
                if len(CLASSIFIERS) > number >= 0:
                    numbers.append(number)
                else:
                    raise ValueError()
            break
        except ValueError:
            print "Please insert a correct list of numbers"

    return [CLASSIFIERS[i] for i in numbers]


def choose_classifier():
    print("Choose a classifier by typing a number:")
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(CLASSIFIERS))

    while True:
        try:
            number = int(raw_input())
            if len(CLASSIFIERS) > number >= 0:
                break
            else:
                raise ValueError()
        except ValueError:
            print "Please insert a correct number"

    return CLASSIFIERS[number]
