import ast

import multiprocessing

from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.features.word_embeddings.keras_word_embedding import *
from src.features.sentence_embeddings.sentence_embeddings import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.common import choose_classifier, LABELS, SENTENCES, CATEGORIES
from src.models.model_testing.grid_search import get_best_from_grid_search_results

if __name__ == "__main__":
    """ Enables user to test chosen classifier by typing sentences interactively"""

    classifier = choose_classifier()
    best_parameters = get_best_from_grid_search_results(classifier)

    if best_parameters is None:
        exit(-1)

    embedding, params = best_parameters
    word_emb_class, sen_emb_class = tuple(embedding.split(","))

    print ("\nEvaluating model for embedding {:s} with params {:s}\n".format(embedding, str(params)))
    params["n_jobs"] = multiprocessing.cpu_count()

    print ("Building word embedding...")
    word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
    word_emb.build(SENTENCES)
    sen_emb = eval(sen_emb_class)()

    print ("Building sentence embedding...")
    sen_emb.build(word_emb, LABELS, SENTENCES)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, LABELS, SENTENCES)

    print ("Building model...")
    clf = classifier(sen_emb, probability=True, **params)
    clf.fit(fb.features, fb.labels)

    print ("Model evaluated!...\n")
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = make_dataset.string_to_words_list(command)
        print map(lambda (i, prob): "{:s}: {:4.2f}%".format(CATEGORIES[i], 100.0*prob),
                  enumerate(clf.predict_proba(sentence)))

