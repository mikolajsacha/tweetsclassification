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
import inflect
import nltk
import numpy as np

inflect_eng = inflect.engine()

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


def preprocess_sentence(sentence):
    """Adjusts sentence by filtering words and correcting some common issues """

    #  twitter users often forget to put space before special words
    sentence = sentence.replace('http', ' http').replace('www', ' www').replace('@', ' @').replace('#', ' #')

    new_sentence = []
    alpha_numeric = re.compile("[^a-z0-9]")

    sentence = ' '.join(filter(lambda w: not (w.startswith('@') or w.startswith('&') or
                                              w.startswith('http') or w.startswith('www')), sentence.split()))
    for w in alpha_numeric.sub(' ', sentence).split():
        if w.isdigit():  # convert numbers to words using inflect package
            new_sentence.append(inflect_eng.number_to_words(int(w)))
        elif not w.isalpha() or w in cached_stopwords or len(w) < 3:
            continue
        else:
            new_sentence.append(w)

    return new_sentence


def string_to_words_list(sentence):
    words = preprocess_sentence(sentence.lower())  # filter words and correct some common issues in sentence
    return words


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
            category, sentence = line.split(' ', 1)
            keywords = string_to_words_list(sentence)
            output_data_file.write("{0} {1}\n".format(category, ','.join(keywords)))

    print "Processed data written to " + output_file_path


def read_dataset(data_file_path, data_info):
    data_set_size = data_info["Size"]

    labels = np.empty(data_set_size, dtype=np.uint8)
    sentences = np.empty(data_set_size, dtype=object)

    count = 0
    for line in open(data_file_path, 'r'):
        label, rest = line.split(' ', 1)
        sentence = string_to_words_list(rest)
        if len(sentence) > 0:
            sentences[count] = sentence
            labels[count] = int(label)
            count += 1

    labels = labels[:count]
    sentences = sentences[:count]

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
