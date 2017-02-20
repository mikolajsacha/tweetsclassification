from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.data import dataset
from src.features.sentence_embeddings import sentence_embeddings
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.keras_neural_network import KerasNeuralNetworkAlgorithm
from src.models.algorithms.sklearn_neural_network import MLPAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.nearest_neighbors_algorithm import NearestNeighborsAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm


def log_range(min_ten_power, max_ten_power, base=10):
    return (base ** i for i in xrange(min_ten_power, max_ten_power))


DATA_FOLDER = "gathered_dataset"
FOLDS_COUNT = 5
TRAINING_SET_SIZE = 0.80

# FAKE classifier just to behave same as other classifiers from sklearn
class KerasNeuralNetworkClassifier(object):
    pass

# CLASSIFIERS_PARAMS = [(SVC, {"C": list(log_range(-1, 8)), "gamma": list(log_range(-10, 1))}),
#                       (RandomForestClassifier, {"criterion": ["gini", "entropy"],
#                                                "min_samples_split": [2, 5, 10, 15],
#                                                "min_samples_leaf": [1, 5, 10, 15],
#                                                 "n_estimators": [1, 5, 10, 15, 20, 35, 50, 100],
#                                                "max_features": [None, "sqrt"]}),
#                       (MLPClassifier, {"alpha": list(log_range(-5, -2)),
#                                                 "learning_rate": ["constant", "adaptive"],
#                                                 "activation": ["identity", "logistic", "tanh", "relu"],
#                                                 "hidden_layer_sizes": [(100,), (100, 50)]}),
#                       (KNeighborsClassifier, {'n_neighbors': [1, 2, 3, 4, 7, 10, 12, 15, 30, 50, 75, 100, 150, 200],
#                                               'weights': ['uniform', 'distance']}),
#                       (KerasNeuralNetworkClassifier, {'nb_epoch': [64, 128, 256, 512, 1024, 2048, 4096, 8192],
#                                                       'batch_size':  [1, 2, 3, 4, 8]})
#                       ]

CLASSIFIERS_PARAMS = [(SVC, {"C": list(log_range(3, 12)), "gamma": list(log_range(-7, -2))}),
                       (RandomForestClassifier, {"max_depth": [None, 1, 5, 10, 20, 50],
                                                 "criterion": ["gini", "entropy"],
                                                 "max_features": ["auto", "sqrt", "log2"],
                                                 "n_estimators": [1000]}),
                      (MLPClassifier, {"alpha": list(log_range(-4, -2)),
                                       "learning_rate": ["constant", "adaptive"],
                                       "activation": ["logistic", "relu"],
                                       "hidden_layer_sizes": [(100, 50)]}),
                      (KNeighborsClassifier, {'n_neighbors': [1, 2, 3, 4, 7, 10, 12, 14, 16, 18,
                                                              20, 25, 30, 50, 75, 100, 150, 200],
                                              'p' : [1, 2, 3, 4, 5],
                                              'weights': ['uniform', 'distance']}),
                      # (KerasNeuralNetworkClassifier, {'nb_epoch': [4048],
                      #                                'batch_size':  [4]})
                      ]

CLASSIFIERS_WRAPPERS = {
    KNeighborsClassifier: NearestNeighborsAlgorithm,
    SVC: SvmAlgorithm,
    RandomForestClassifier: RandomForestAlgorithm,
    MLPClassifier: MLPAlgorithm,
    KerasNeuralNetworkClassifier: KerasNeuralNetworkAlgorithm
}

CLASSIFIERS = [c[0] for c in CLASSIFIERS_PARAMS]

SENTENCE_EMBEDDINGS = [
    # sentence_embeddings.ConcatenationEmbedding,
    sentence_embeddings.SumEmbedding
]

WORD_EMBEDDINGS = [
    # (Word2VecEmbedding, ['google/GoogleNews-vectors-negative300.bin', 300]),
    # (GloveEmbedding, ['glove_twitter/glove.twitter.27B.200d.txt', 200]),
    (GloveEmbedding, ['glove_twitter/glove.twitter.27B.200d.txt', 200]),
]

EXTERNAL_DATA_PATH = dataset.get_external_data_path(DATA_FOLDER)
PROCESSED_DATA_PATH = dataset.get_processed_data_path(DATA_FOLDER)
DATA_INFO = dataset.read_data_info(dataset.get_data_set_info_path(DATA_FOLDER))
CATEGORIES = DATA_INFO['Categories']
DATA_SIZE = DATA_INFO['Size']

CATEGORIES_COUNT = len(CATEGORIES)
KerasNeuralNetworkAlgorithm.categories_count = CATEGORIES_COUNT

LABELS, SENTENCES = dataset.read(EXTERNAL_DATA_PATH, DATA_INFO)

def choose_multiple_classifiers():
    print("Choose classifiers by typing a list of their indices (e.g: '0,2'):")
    classifier_wrappers = [CLASSIFIERS_WRAPPERS[classifier] for classifier in CLASSIFIERS]
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(classifier_wrappers))
    numbers = []
    while True:
        try:
            str_numbers = raw_input().replace(" ", "").split(',')
            for str_number in str_numbers:
                number = int(str_number)
                if len(classifier_wrappers) > number >= 0:
                    numbers.append(number)
                else:
                    raise ValueError()
            break
        except ValueError:
            print "Please insert a correct list of numbers"

    return [classifier_wrappers[i] for i in numbers]


def choose_classifier():
    print("Choose a classifier by typing a number:")
    classifier_wrappers = [CLASSIFIERS_WRAPPERS[classifier] for classifier in CLASSIFIERS]
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(classifier_wrappers))

    while True:
        try:
            number = int(raw_input())
            if len(classifier_wrappers) > number >= 0:
                break
            else:
                raise ValueError()
        except ValueError:
            print "Please insert a correct number"

    return classifier_wrappers[number]
