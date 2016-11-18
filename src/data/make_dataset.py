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
import numpy as np
from nltk.corpus import stopwords

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


def get_max_sentence_length(data_file_path):
    """
    :param data_file_path: absolute path to data file
    :type data_file_path: string (path to a file)
    :return length (in words count) of the longest sentence from data set
    """

    def sentence_length(line):
        keywords = re.compile('[a-zA-Z]+').findall(line)  # get all words as a list
        return len(filter_words(keywords))  # filter out unnecessary words and count them

    return reduce(lambda acc, x: max(acc, x), (sentence_length(line) for line in open(data_file_path, 'r')), 0)


def get_max_word_length(data_file_path):
    """
    Returns length of the longest word in file. Used to optimize arrays of strings.
    :param data_file_path: absolute path to data file
    :type data_file_path: string (path to a file)
    :return length of the longest word in data set
    """

    def max_word_length(line):
        keywords = re.compile('[a-zA-Z]+').findall(line)  # get all words as a list
        return reduce(lambda acc, word: max(acc, len(word)), keywords, 0)

    return reduce(lambda acc, x: max(acc, x), (max_word_length(line) for line in open(data_file_path, 'r')), 0)


def string_to_words_list(sentence, vector_length):
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
            keywords = string_to_words_list(line, vector_length)
            output_data_file.write("{0} {1}\n".format(category, ','.join(keywords)))

    print "Processed data written to " + output_file_path


def read_dataset(data_file_path, data_info):
    vector_length = get_max_sentence_length(data_file_path)
    max_word_length = get_max_word_length(data_file_path)
    data_set_size = data_info["Size"]

    labels = np.empty(data_set_size, dtype=np.uint8)
    sentences = np.empty((data_set_size, vector_length), dtype='|S{:d}'.format(max_word_length))

    for i, line in enumerate(open(data_file_path, 'r')):
        label, rest = line.split(' ', 1)
        labels[i] = int(label)
        sentences[i] = string_to_words_list(rest, vector_length)

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
            length = 0
            while length < 1:
                length_input = raw_input("Type desired vector length (integer greater than zero) or 'auto' " +
                                         "to use the length of the longest sentence: ")
                if length_input.lower() == "auto":
                    length = get_max_sentence_length(input_file_path)
                    print("Use {0} as the vector length".format(length))
                    make_dataset(input_file_path, output_file_path, length)
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
