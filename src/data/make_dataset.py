"""
Contains methods for preprocessing given dataset so they can be translated into more accurate model.
Data sets are taken from folder "data/datasetname/external".
After processing, data sets are saved in folder "data/datasetname/processed".

As of now, processing involves only filtering words longer than one character
and trimming/extending word vectors to given length.
"""

import re
import os
from nltk.corpus import stopwords
import numpy as np

cached_stopwords = set(stopwords.words("english"))


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


def get_max_sentence_length(data_folder):
    """
    :param data_folder: name of data folder, e.g. 'dataset1'
    :type data_folder: string (path to a folder)
    :return length (in words count) of the longest sentence from data set
    """
    data_file_path = get_external_data_path(data_folder)

    def sentence_length(line):
        keywords = re.compile('[a-zA-Z]+').findall(line)  # get all words as a list
        return len(filter_words(keywords))  # filter out unnecessary words and count them

    return reduce(lambda acc, x: max(acc, x), (sentence_length(line) for line in open(data_file_path, 'r')), 0)


def sentence_to_word_vector(sentence, vector_length):
    keywords = re.compile('[a-zA-Z]+').findall(sentence)  # get all words as a list
    keywords = filter_words(keywords)  # filter out unnecessary words
    keywords = keywords[:vector_length]  # trim keywords length
    keywords = map(lambda word: word.lower(), keywords)  # map words to lowercase

    #  if vector is too short, fill with empty words
    for i in xrange(len(keywords), vector_length):
        keywords.append('')

    return keywords


def make_dataset(data_file_path, output_file_path, vector_length):
    """
    Generates files with data represented as vectors of words of fixed length.

    Words shorter than required length will be extended by empty words.
    Words that are too long will be trimmed.

    :param data_file_path: relative path to file with data set
    :param output_file_path: relative path to which processed data should be written
    :param vector_length: desired length of each data set entry (e.g. 5)
    :type data_file_path: string (path to data file)
    :type output_file_path: int (non-negative)
    """

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, 'w') as output_data_file:
        for line in open(data_file_path, 'r'):
            category = line.split(' ', 1)[0]
            keywords = sentence_to_word_vector(line, vector_length)
            output_data_file.write("{0} {1}\n".format(category, ','.join(keywords)))

    print "Processed data written to " + output_file_path


def read_dataset(data_folder):
    data_file_path = get_external_data_path(data_folder)
    vector_length = get_max_sentence_length(data_folder)
    labels = []
    sentences = []
    for line in open(data_file_path, 'r'):
        label, rest = line.split(' ', 1)
        labels.append(int(label))
        sentences.append(sentence_to_word_vector(rest, vector_length))
    return labels, sentences


if __name__ == "__main__":
    """
    Main method allows to generate processed data sets in interactive mode.
    """
    while True:
        command = raw_input("Type data set folder name to generate data set or 'quit' to quit script: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break

        input_file_path = get_external_data_path(command)

        if not os.path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)

        else:
            length = 0
            while length < 1:
                length_input = raw_input("Type desired vector length (integer greater than zero) or 'auto' " +
                                         "to use the length of the longest sentence: ")
                if length_input.lower() == "auto":
                    length = get_max_sentence_length(command)
                    print("Use {0} as the vector length".format(length))
                    make_dataset(input_file_path, get_processed_data_path(command), length)
                    break;

                try:
                    length = int(length_input)
                except ValueError:
                    print "Vector length must be an integer"
                    continue

                if length < 1:
                    print "Vector length must be greather than zero"
                    continue

                make_dataset(input_file_path, get_processed_data_path(command), length)
