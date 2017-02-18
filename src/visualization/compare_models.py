import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # do not remove this import
import matplotlib.patches as mpatches
import os

from src.common import CLASSIFIERS_WRAPPERS, CLASSIFIERS, DATA_FOLDER
from src.models.model_testing.grid_search import get_best_from_grid_search_results_for_classifier, \
    get_grid_search_results_path
from src.visualization.save_visualization import save_current_plot


def get_average_grid_search_result_for_classifier(classifier):
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        return None

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to get some results."
        return None

    results_sum = 0.0
    results_count = 0.0

    for line in open(summary_file_path, 'r'):
        split_line = tuple(line.split(";"))
        result = float(split_line[-1])
        results_sum += result
        results_count += 1.0

    return round(results_sum/results_count, 2)



def get_available_grid_search_results():
    # returns best and average results for each classifier with existing grid search result file
    results_for_classifiers = []
    classifier_wrappers = [CLASSIFIERS_WRAPPERS[classifier] for classifier in CLASSIFIERS]
    for classifier_class in classifier_wrappers:
        results = get_best_from_grid_search_results_for_classifier(classifier_class, include_result=True)
        if results is not None:
            average = get_average_grid_search_result_for_classifier(classifier_class)
            results_for_classifiers.append((classifier_class, results + (average,)))

    return results_for_classifiers


def compare_models_bar_chart(best_results_for_models):
    # best_results = [(cls_class, (word_emb_class, word_emb_params, sen_emb_class, params, best_result, avg_result)]

    # sort results by performance
    best_results_for_models.sort(key=lambda (cls, params): params[4])

    fig, ax = plt.subplots()
    N , width = len(best_results_for_models), 0.35
    ind = np.arange(N)

    classifier_classes = [clf.__name__ for clf, _ in best_results_for_models]
    average_performances = [params[5] for _, params in best_results_for_models]
    max_performances = [params[4] for _, params in best_results_for_models]

    rects1 = ax.bar(ind, max_performances, width, color='b')
    rects2 = ax.bar(ind, average_performances, width, color='g')
    plt.xticks(ind, classifier_classes)

    avg_legend = mpatches.Patch(color='g', label="Average performance")
    max_legend = mpatches.Patch(color='b', label="Maximum performance")

    plt.legend(handles=[avg_legend, max_legend])

    plt.title('Comparison of performance of tested models')
    plt.ylabel('Cross-validation results')

    # Attach a text label above each bar displaying its value
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height, "{:4.2f}%".format(height),
                    ha='center', va='bottom')

    save_current_plot('models_comparison.svg')
    plt.show()

if __name__ == "__main__":
    best_results_for_models =  get_available_grid_search_results()

    for classifier_class, parameters in best_results_for_models:
        word_emb_class, word_emb_params, sen_emb_class, params, best_result, avg_result = parameters

        print ("\n{0}: Best result: {1}%, Average result: {2}%".
               format(classifier_class.__name__, best_result, avg_result))
        print ("For embeddings: {0}({1}), {2}".format(word_emb_class.__name__,
                                                      ', '.join(map(str, word_emb_params)),
                                                      sen_emb_class.__name__))
        print ("And for parameters: {0}".format(str(params)))

    compare_models_bar_chart(best_results_for_models)



