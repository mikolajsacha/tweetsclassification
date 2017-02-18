import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # do not remove this import
import matplotlib.patches as mpatches
import os

from src.common import CLASSIFIERS_WRAPPERS, CLASSIFIERS, DATA_FOLDER, SENTENCE_EMBEDDINGS
from src.models.model_testing.grid_search import get_best_from_grid_search_results_for_classifier, \
    get_grid_search_results_path
from src.visualization.save_visualization import save_current_plot


def get_grid_search_results_by_sentence_embeddings(sen_embeddings):
    classifier_wrappers = [CLASSIFIERS_WRAPPERS[classifier] for classifier in CLASSIFIERS]

    results_for_classifiers = []

    for classifier_class in classifier_wrappers:
        summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier_class)

        if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)) or \
                os.stat(summary_file_path).st_size == 0:
            continue

        print("Found Grid Search results in " + summary_file_path.split("..")[-1])

        results_for_classifiers.append((classifier_class, []))

        for sen_emb in sen_embeddings:
            max_result = 0.0
            results_sum = 0.0
            results_count = 0.0

            for line in open(summary_file_path, 'r'):
                split_line = tuple(line.split(";"))
                params = split_line[:-1]
                if params[2] == sen_emb:
                    result = float(split_line[-1])
                    if result > max_result:
                        max_result = result
                    results_sum += max_result
                    results_count += 1

            avg_result = round(results_sum/results_count, 2) if results_count > 0 else 0

            results_for_classifiers[-1][1].append((sen_emb, max_result, avg_result))

    return results_for_classifiers


def compare_sentence_embeddings_bar_chart(results_for_classifiers):
    # sort results by performance

    fig, ax = plt.subplots()
    N , width = len(results_for_classifiers), 0.35
    ind = np.arange(N)

    max_colors = ['#641E16', '#154360', "#7D6608", "#7B7D7D"]
    avg_colors = ['#C0392B', '#2980B9', "#F1C40F", "#ECF0F1"]

    classifier_classes = [clf.__name__ for clf, _ in results_for_classifiers]
    sen_embeddings = [sen_emb for sen_emb, max_r, avg_r in results_for_classifiers[0][1]]

    legend_handles = []

    for i, sen_embedding in enumerate(sen_embeddings):
        max_performances = [params[i][1] for clf, params in results_for_classifiers]
        average_performances = [params[i][2] for clf, params in results_for_classifiers]
        rects1 = ax.bar(ind + i * width, max_performances, width, color=max_colors[i])
        rects2 = ax.bar(ind + i * width, average_performances, width, color=avg_colors[i])

        # Attach a text label above each bar displaying its value
        for shift, rects in [(0, rects1), (-2, rects2)]:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + shift, "{:4.2f}%".format(height),
                        ha='center', va='bottom')

        # Add description to legend
        legend_handles.append(mpatches.Patch(color=max_colors[i], label=sen_embedding + " maximum performance"))
        legend_handles.append(mpatches.Patch(color=avg_colors[i], label=sen_embedding + " average performance"))


    plt.xticks(ind + width * (len(sen_embeddings) - 1) / 2.0, classifier_classes)

    plt.legend(handles=legend_handles)

    plt.title('Comparison of performance of Sentence Embeddings')
    plt.ylabel('Cross-validation results')

    save_current_plot('sentence_embeddings_comparison.svg')
    plt.show()

if __name__ == "__main__":
    sen_embeddings = [sen_emb.__name__ for sen_emb in SENTENCE_EMBEDDINGS]

    grid_search_results = get_grid_search_results_by_sentence_embeddings(sen_embeddings)
    compare_sentence_embeddings_bar_chart(grid_search_results)



