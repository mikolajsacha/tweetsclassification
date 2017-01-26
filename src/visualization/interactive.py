import ast
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.features.word_embeddings.keras_word_embedding import *
from src.features.sentence_embeddings.sentence_embeddings import *
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.configuration import DATA_FOLDER
from src.models.model_testing.grid_search import get_grid_search_results_path

if __name__ == "__main__":
    """ Enables user to test chosen classifier by typing sentences interactively"""

    classifiers = [SvmAlgorithm, RandomForestAlgorithm, NeuralNetworkAlgorithm]

    number = "x"
    print("Choose a classifier to test by typing a number:")
    print "\n".join("{0} - {1}".format(i, clf.__name__) for i, clf in enumerate(classifiers))

    while True:
        try:
            number = int(raw_input())
            if len(classifiers) > number >= 0:
                break
            else:
                raise ValueError()
        except ValueError:
            print "Please insert a correct number"

    classifier = classifiers[number]

    data_path = make_dataset.get_processed_data_path(DATA_FOLDER)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(DATA_FOLDER))
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        exit(-1)

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to gest some results."
        exit(-1)

    max_result = 0.0
    best_parameters = None

    print("Found Grid Search results in " + summary_file_path.split("..")[-1])
    for line in open(summary_file_path, 'r'):
        embedding, params, result = tuple(line.split(";"))
        result = float(result)
        if result > max_result:
            max_result = result
            best_parameters = embedding, ast.literal_eval(params)

    labels, sentences = make_dataset.read_dataset(data_path, data_info)
    embedding, params = best_parameters
    word_emb_class, sen_emb_class = tuple(embedding.split(","))

    print ("\nEvaluating model for embedding {:s} with params {:s}\n".format(embedding, str(params)))

    print ("Building word embedding...")
    word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
    word_emb.build(sentences)
    sen_emb = eval(sen_emb_class)()

    print ("Building sentence embedding...")
    sen_emb.build(word_emb, labels, sentences)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, labels, sentences)

    print ("Building model...")
    clf = classifier(sen_emb, probability=True, **params)
    clf.fit(fb.features, fb.labels)

    print ("Model evaluated!...\n")
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = make_dataset.string_to_words_list(command)
        print map(lambda (i, prob): "{:s}: {:4.2f}%".format(data_info["Categories"][i], 100.0*prob),
                  enumerate(clf.predict_proba(sentence)))

