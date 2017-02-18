import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D # do not remove this import
import matplotlib.patches as mpatches
from matplotlib import cm
from scipy.interpolate import griddata

from src.common import choose_classifier, DATA_FOLDER
from src.models.model_testing.grid_search import get_grid_search_results_path
from src.visualization.save_visualization import save_current_plot


def get_all_grid_searched_parameters(classifier_class):
    """ Returns list of combinations of parameters as a list of pairs (param_dictionary, validation_result) """
    summary_file_path = get_grid_search_results_path(DATA_FOLDER, classifier_class)

    if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
        print "Grid Search summary file does not exist. Please run grid_search.py at first."
        return None

    if os.stat(summary_file_path).st_size == 0:
        print "Grid Search summary file is empty. Please run grid_search.py to get some results."
        return None

    all_parameters = []

    print("Found Grid Search results in " + summary_file_path.split("..")[-1])
    for line in open(summary_file_path, 'r'):
        split_line = tuple(line.split(";"))
        result = float(split_line[-1])
        parameters_str = split_line[:-1]

        word_embedding, word_embedding_params, sentence_embedding, classifier_params = parameters_str

        parameters_dict = ast.literal_eval(classifier_params)
        parameters_dict["Sentence Embedding"] = sentence_embedding
        parameters_dict["Word Embedding"] = "{0}({1})".format(word_embedding, word_embedding_params)
        all_parameters.append((parameters_dict, result))

    return all_parameters


def choose_parameters_to_analyze(parameters_list):
    """ Lets user choose one or two parameters to analyze """
    print "Choose one or two of the following parameters by typing a number or two numbers, e.g. '1' or '3,4': "
    print "\n".join("{0} - {1}".format(i, param_name) for i, param_name in enumerate(parameters_list))

    numbers = []
    max_number = min(2, len(parameters_list))
    while True:
        try:
            str_numbers = raw_input().replace(" ", "").split(',')
            if len(str_numbers) > max_number:
                raise ValueError()
            for str_number in str_numbers:
                number = int(str_number)
                if len(parameters_list) > number >= 0:
                    numbers.append(number)
                else:
                    raise ValueError()
            break
        except ValueError:
            print "Please insert a correct number or two numbers"

    return [parameters_list[i] for i in numbers]


def analyze_single_parameter(parameter, classifier_class, all_parameters_list):
    # count average, max and min performance for each value of parameter

    average_performances = {}
    min_performances = {}
    max_performances = {}

    for parameters, result in all_parameters_list:
        tested_param_value = parameters[parameter]

        if tested_param_value in average_performances:
            average_performances[tested_param_value] += result
            if result < min_performances[tested_param_value]:
                min_performances[tested_param_value] = result
            if result > max_performances[tested_param_value]:
                max_performances[tested_param_value] = result
        else:
            average_performances[tested_param_value] = result
            min_performances[tested_param_value] = result
            max_performances[tested_param_value] = result

    param_values = sorted(average_performances.iterkeys())
    param_values_count = len(param_values)
    tests_count_per_param_value = len(all_parameters_list) / param_values_count

    for param_value in average_performances.iterkeys():
        average_performances[param_value] /= tests_count_per_param_value

    # convert dictionaries to lists sorted by tested param values
    average_performances = [average_performances[key] for key in param_values]
    min_performances = [min_performances[key] for key in param_values]
    max_performances = [max_performances[key] for key in param_values]

    fig, ax = plt.subplots()
    use_log_scale = False

    # if parameter is numerical, plot lines and ask if to use logarithmic scale
    if all(isinstance(x, int) or isinstance(x, float) for x in param_values):
        use_log_answer = raw_input("Use logarithmic scale? [y/n] ").lower()
        use_log_scale = use_log_answer == 'y' or use_log_answer == 'yes'
        if use_log_scale:
            ax.set_xscale('log')
        lines = ax.plot(param_values, average_performances, 'orange', param_values, min_performances, 'r',
                         param_values, max_performances, 'g')

        ax.scatter(param_values, average_performances, c='orange', s=150, marker='*', edgecolors='black')
        ax.scatter(param_values, min_performances, c='red', s=150, marker='*', edgecolors='black')
        ax.scatter(param_values, max_performances, c='green', s=150, marker='*', edgecolors='black')

        plt.setp(lines, linewidth=2, markersize=8)

    # if parameter is non-numerical, plot a bar chart
    else:
        N , width = param_values_count, 0.15
        ind = np.arange(N)
        ax.bar(ind, average_performances, width, color='orange', label='Avg')
        ax.bar(ind + width, min_performances, width, color='r', label='Min')
        ax.bar(ind + 2 * width, max_performances, width, color='g', label='Max')
        plt.xticks(ind + width, param_values)

    avg_legend = mpatches.Patch(color='orange', label="Average performance")
    min_legend = mpatches.Patch(color='r', label="Minimum performance")
    max_legend = mpatches.Patch(color='g', label="Maximum performance")

    plt.legend(handles=[avg_legend, min_legend, max_legend])

    plt.title('{0} performance for different values of {1}'.format(classifier_class.__name__, parameter))

    if use_log_scale:
        plt.xlabel('Values of {0} (logarithmic scale)'.format(parameter))
    else:
        plt.xlabel('Values of {0}'.format(parameter))

    plt.ylabel('Cross-validation results')

    save_current_plot('parameters_{0}_{1}.svg'.format(classifier_class.__name__, parameter))
    plt.show()


def analyze_two_parameters(parameter1, parameter2, classifier_class, all_parameters_list):
    # count max performance for each combinations of value of parameter1 and parameter2

    max_performances = {}

    for parameters, result in all_parameters_list:
        tested_param1_value = parameters[parameter1]
        tested_param2_value = parameters[parameter2]

        tested_tuple = (tested_param1_value, tested_param2_value)

        if tested_tuple in max_performances:
            if result > max_performances[tested_tuple]:
                max_performances[tested_tuple] = result
        else:
            max_performances[tested_tuple] = result

    param1_values = sorted([p1 for p1, p2 in max_performances.iterkeys()])
    param2_values = sorted([p2 for p1, p2 in max_performances.iterkeys()])

    # plot makes sense only if parameters ale numerical
    if not all(isinstance(x, int) or isinstance(x, float) for x in param1_values) or \
       not all(isinstance(x, int) or isinstance(x, float) for x in param2_values):
        print "Tested parameters must be numerical. Non-numerical paramateres can be analyzed only individually"
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    zv = np.empty((len(param1_values), len(param2_values)))
    for i, param1 in enumerate(param1_values):
        for j, param2 in enumerate(param2_values):
            zv[i, j] = max_performances[(param1, param2)]


    points = np.zeros((len(param1_values) * len(param2_values), 2))
    values = np.zeros((len(param1_values) * len(param2_values)))

    param1_points = param1_values[:]
    param2_points = param2_values[:]

    use_log_answer = raw_input("Use logarithmic scale for {0}? [y/n] ".format(parameter1)).lower()
    use_log_scale1 = use_log_answer == 'y' or use_log_answer == 'yes'
    if use_log_scale1:
        param1_points = np.log2(param1_values)

    use_log_answer = raw_input("Use logarithmic scale for {0}? [y/n] ".format(parameter2)).lower()
    use_log_scale2 = use_log_answer == 'y' or use_log_answer == 'yes'
    if use_log_scale2:
        param2_points = np.log2(param2_values)


    # interpolate for better visual effect
    point_i = 0
    for i, param1_val in enumerate(param1_values):
        for j, param2_val in enumerate(param2_values):
            points[point_i] = [param1_points[i], param2_points[j]]
            values[point_i] = max_performances[(param1_val, param2_val)]
            point_i += 1

    grid_size = 20
    grid_x, grid_y = np.meshgrid(np.linspace(param1_points[0], param1_points[-1], num=grid_size),
                                 np.linspace(param2_points[0], param2_points[-1], num=grid_size))
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    for i in xrange(grid_z.shape[0]):
        for j in xrange(grid_z.shape[1]):
            if grid_z[i,j] > 100:
                grid_z[i, j] = 100

    ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.coolwarm, linewidth=0, alpha=0.8)


    # scatter real points
    xs_and_ys = list(itertools.product(param1_points, param2_points))
    xs = [x for x, y in xs_and_ys]
    ys = [y for x, y in xs_and_ys]
    zs = [max_performances[(x,y)] for (x, y) in itertools.product(param1_values, param2_values)]

    ax.scatter(xs, ys, zs, s=5)


    plt.title('{0} performance for different values of {1} and {2}'
              .format(classifier_class.__name__, parameter1, parameter2))

    if use_log_scale1:
        ax.set_xlabel('Values of {0} (logarithmic scale: 2^))'.format(parameter1))
    else:
        ax.set_xlabel('Values of {0}'.format(parameter1))

    if use_log_scale2:
        ax.set_ylabel('Values of {0} (logarithmic scale: 2^))'.format(parameter2))
    else:
        ax.set_ylabel('Values of {0}'.format(parameter2))


    ax.set_zlabel('Cross-validation results')

    save_current_plot('parameters_{0}_{1}_and_{2}.svg'.format(classifier_class.__name__, parameter1, parameter2))
    plt.show()


if __name__ == "__main__":
    classifier_class = choose_classifier()
    parameters_list = get_all_grid_searched_parameters(classifier_class)
    if not parameters_list:
        exit(-1)
    tested_parameters = list(parameters_list[0][0].iterkeys())
    parameters_to_analyze = choose_parameters_to_analyze(tested_parameters)

    if len(parameters_to_analyze) == 1:
        analyze_single_parameter(parameters_to_analyze[0], classifier_class, parameters_list)

    elif len(parameters_to_analyze) == 2:
        analyze_two_parameters(parameters_to_analyze[0], parameters_to_analyze[1], classifier_class, parameters_list)





