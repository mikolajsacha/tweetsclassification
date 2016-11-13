import re
import os


def get_external_data_path(data_folder):
    return '../../data/{0}/external/training_set.txt'.format(data_folder)


def get_processed_data_path(data_folder):
    directory = '../../data/{0}/processed'.format(data_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory + "/training_set.txt"


def make_dataset(data_file_path, output_file_path, vector_length):
    """
    Generates files with data represented as vectors of words of fixed length.

    Words shorter than required length will be extended by empty words.
    Words that are too long will be trimmed.

    :param data_file_path: relative path to file with data set
    :param output_file_path: relative path to which processed data should be written
    :param vector_length: desired length of each data set entry (e.g. 5)
    :type arg1: string (path to data file)
    :type arg2: int (non-negative)
    """

    with open(output_file_path, 'w') as output_data_file:
        for line in open(data_file_path, 'r'):
            category = line.split(' ', 1)[0]
            keywords = re.compile('[a-zA-Z]+').findall(line)  # get all words as a list
            keywords = filter(lambda word: len(word) > 1, keywords)  # filter out 1 character words
            keywords = keywords[:vector_length] # trim keywords length
            keywords = map(lambda word: word.lower(), keywords)  # map words to lowercase

            #  if vector is too short, fill with empty words
            for i in xrange(len(keywords), vector_length):
                keywords.append('')

            output_data_file.write("{0} {1}\n".format(category, ','.join(keywords)))

    print "Processed data written to " + output_file_path

if __name__ == "__main__":
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
                length_input = raw_input("Type desired vector length (integer greater than zero): ")
                try:
                    length = int(length_input)
                except ValueError:
                    print "Vector length must be an integer"
                    continue

                if length < 1:
                    print "Vector length must be greather than zero"
                    continue

                make_dataset(input_file_path, get_processed_data_path(command), length)
