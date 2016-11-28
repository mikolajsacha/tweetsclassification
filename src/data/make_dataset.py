"""
Contains methods for preprocessing given dataset so they can be translated into more accurate model.
Data sets are taken from folder "data/datasetname/external".
After processing, data sets are saved in folder "data/datasetname/processed".

As of now, processing involves only filtering words longer than one character
and trimming/extending word vectors to given length.
"""

import re
import os
import json
import nltk
import numpy as np

while True:
    try:
        cached_stopwords = set(nltk.corpus.stopwords.words("english"))
        break
    except LookupError:
        print ("Please download 'stopwords' using NTLK download manager")
        nltk.download()


def get_data_set_info_path(data_folder):
    """
    :param data_folder: name of data folder, e.g. 'dataset1'
    :return: absolute path to JSON file containing information about data set
    """
    return os.path.join(os.path.dirname(__file__), '..\\..\\data\\{0}\\external\\data_info.json'.format(data_folder))


def get_external_data_path(data_folder):
    """
    :param data_folder: name of data folder, e.g. 'dataset1'
    :return: absolute path to external data set file for this folder name
    """
    return os.path.join(os.path.dirname(__file__), '..\\..\\data\\{0}\\external\\training_set.txt'.format(data_folder))


def get_processed_data_path(data_folder):
    """
    :param data_folder: name of data folder, e.g. 'dataset1'
    :type data_folder: string (path to a folder)
    :return: absolute path to processed data set file for this folder name
    """
    return os.path.join(os.path.dirname(__file__), '..\\..\\data\\{0}\\processed\\training_set.txt'.format(data_folder))


def filter_words(sentence):
    """
    Filters unnecessary words from given sentence and return new sentence
    :param sentence: a list of words
    :type sentence: list of strings
    :return sentence with unnecessary words filtered out
    """
    # More filtering can be implemented in the future
    return filter(lambda word: word not in cached_stopwords, sentence)  # filter out stop words


def string_to_words_list(sentence):
    keywords = re.compile('[a-zA-Z]+').findall(sentence)  # get all words as a list
    keywords = filter_words(keywords)  # filter out unnecessary words
    keywords = map(lambda word: word.lower(), keywords)  # map words to lowercase
    return keywords


def make_dataset(data_file_path, output_file_path):
    """
    Generates files with data represented as vectors of words of fixed length.

    Words shorter than required length will be extended by empty words.
    Words that are too long will be trimmed.

    :param data_file_path: relative path to file with data set
    :param output_file_path: relative path to which processed data should be written
    :type data_file_path: string (path to data file)
    :type output_file_path: int (non-negative)
    """

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as output_data_file:
        for line in open(data_file_path, 'r'):
            category = line.split(' ', 1)[0]
            keywords = string_to_words_list(line)
            output_data_file.write("{0} {1}\n".format(category, ','.join(keywords)))

    print "Processed data written to " + output_file_path


def read_dataset(data_file_path, data_info):
    data_set_size = data_info["Size"]

    labels = np.empty(data_set_size, dtype=np.uint8)
    sentences = np.empty(data_set_size, dtype=object)

    for i, line in enumerate(open(data_file_path, 'r')):
        label, rest = line.split(' ', 1)
        labels[i] = int(label)
        sentences[i] = string_to_words_list(rest)

    labels.flags.writeable = False
    sentences.flags.writeable = False

    return labels, sentences


def read_data_info(data_set_info_path):
    with open(data_set_info_path) as data_file:
        return json.load(data_file)


if __name__ == "__main__":
    """
    Main method allows to generate processed data sets in interactive mode.
    """
    while True:
        command = raw_input("Type data set folder name to generate data set or 'quit' to quit script: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break

        input_file_path = get_external_data_path(command)
        output_file_path = get_processed_data_path(command)

        if not os.path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)

        else:
            make_dataset(input_file_path, output_file_path)
