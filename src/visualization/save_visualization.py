import os

import matplotlib.pyplot as plt


def get_figures_path():
    """
    :return: absolute path to the location where figures are stored (as images/pdfs etc)
    """
    return os.path.join(os.path.dirname(__file__), '..\\..\\reports\\figures')


def save_current_plot(filename):
    path = os.path.join(get_figures_path(), filename)
    plt.savefig(path)
    print "Plot saved in " + path.split("..")[-1]
