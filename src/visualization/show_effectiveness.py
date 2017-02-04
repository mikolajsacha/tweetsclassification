import ast

import multiprocessing

from src.features import build_features
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.common import choose_classifier, SENTENCES, LABELS
from src.models.model_testing.grid_search import get_best_from_grid_search_results
from src.models.model_testing.validation import test_cross_validation

if __name__ == "__main__":
    """ Shows which of sentences are guessed wrong for our best classifiers"""
    classifier = choose_classifier()
    best_parameters = get_best_from_grid_search_results(classifier)

    if best_parameters is None:
        exit(-1)

    embedding, params = best_parameters
    _, sen_emb_class = tuple(embedding.split(","))

    print ("\nEvaluating model for embedding {:s} with params {:s}\n".format(embedding, str(params)))
    params["n_jobs"] = multiprocessing.cpu_count()

    word_emb = Word2VecEmbedding('google/GoogleNews-vectors-negative300.bin', 300)
    word_emb.build()
    sen_emb = eval(sen_emb_class)()
    fb = build_features.FeatureBuilder()
    folds_count = 5

    validation_results = test_cross_validation(LABELS, SENTENCES, word_emb, sen_emb, fb,
                                               classifier, folds_count, verbose=True, include_wrong_sentences=True)

    for i, (success_rate, wrong_sentences) in enumerate(validation_results):
        print("\nResult in fold {:d}: {:4.2f}%. Wrong sentences were:".format(i + 1, success_rate * 100))
        print("\n".join("{0} | Real label: {1} | Predicted label: {2}".format(*sent) for sent in wrong_sentences))
