from src.data import dataset
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.models.algorithms.sklearn_neural_network import MLPAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.common import choose_classifier, LABELS, SENTENCES, CATEGORIES
from src.models.model_testing.grid_search import get_best_from_grid_search_results

def interactive_test(clf):
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = dataset.string_to_words_list(command)
        print map(lambda (i, prob): "{:s}: {:4.2f}%".format(CATEGORIES[i], 100.0*prob),
                  enumerate(clf.predict_proba(sentence)))


if __name__ == "__main__":
    """ Enables user to test chosen classifier by typing sentences interactively"""

    classifier = choose_classifier()
    best_parameters = get_best_from_grid_search_results(classifier)

    if best_parameters is None:
        exit(-1)

    word_emb_class, word_emb_params, sen_emb_class, params = best_parameters

    print ("\nEvaluating model for word embedding: {:s}({:s}), sentence embedding: {:s} \nHyperparameters {:s}\n"
           .format(word_emb_class.__name__, ', '.join(map(str, word_emb_params)), sen_emb_class.__name__, str(params)))
    params["n_jobs"] = -1 # use multi-threading

    print ("Building word embedding...")
    word_emb = word_emb_class(*word_emb_params)
    word_emb.build()

    print ("Building sentence embedding...")
    sen_emb = sen_emb_class()
    sen_emb.build(word_emb)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, LABELS, SENTENCES)

    print ("Building model...")
    clf = classifier(sen_emb, probability=True, **params)
    clf.fit(fb.features, fb.labels)

    print ("Model evaluated!...\n")
    interactive_test(clf)

