import ast
from matplotlib import gridspec
from src.features.sentence_embeddings.sentence_embeddings import *
from src.features import build_features
from src.features.word_embeddings.word2vec_embedding import *
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from sklearn.model_selection import StratifiedKFold
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D  # do not remove this import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if __name__ == "__main__":
    """ Enables user to test SVM by typing sentences interactively"""
    data_folder = "dataset3_reduced"
    data_path = make_dataset.get_processed_data_path(data_folder)
    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(data_folder))
    summary_file_path = get_grid_search_results_path(data_folder, SvmAlgorithm)

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
    svm = SvmAlgorithm(sen_emb, probability=True, **params)
    svm.fit(fb.features, fb.labels)

    print ("Model evaluated!...\n")
    while True:
        command = raw_input("Type sentence to test model or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        sentence = make_dataset.string_to_words_list(command)
        print map(lambda (i, prob): "{:s}: {:4.2f}%".format(data_info["Categories"][i], 100.0*prob),
                  enumerate(svm.predict_proba(sentence)))

