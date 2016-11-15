"""
Contains class representing Word2Vec embedding, implementing IWordEmbedding interface
"""
from iword_embedding import IWordEmbedding
from gensim.models import Word2Vec
from src.data.make_dataset import get_processed_data_path
from os import path, makedirs


class Word2VecEmbedding(IWordEmbedding):
    def __init__(self):
        # for start build an empty model
        self.model = Word2Vec([['']], size=0, min_count=1)

    def load(self, file_path):
        self.model = Word2Vec.load(file_path)
        self.model.init_sims(replace=True)  # trim unneeded model memory

    def save(self, data_folder):
        # save in word2vec format
        output_path = self.get_model_data_path(data_folder)
        if not path.exists(path.dirname(output_path)):
            makedirs(path.dirname(output_path))
        self.model.save(output_path)

        # save in human-readable format
        with open(output_path + ".txt", 'w') as f:
            for word in self.model.vocab:
                vector = ','.join(map(lambda val: str(val), self.model[word]))
                f.write("{0} {1}\n".format(word.rstrip(), vector))

    def build(self, sentences, vector_length):
        self.model = Word2Vec(sentences, size=vector_length, min_count=1)
        self.model.init_sims(replace=True)  # trim unneeded model memory

    def __getitem__(self, word):
        return self.model[word]

    def get_model_data_path(self, data_folder):
        return IWordEmbedding.get_model_data_path(data_folder) + '/word2vec'


if __name__ == "__main__":
    """
    Main method allows to interactively build and test Word2Vec model
    """

    while True:
        command = raw_input("Type data set folder name to build Word2Vec embedding: ")
        input_file_path = "../" + get_processed_data_path(command)

        if not path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)
        else:
            break

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

    model = Word2VecEmbedding()
    model.build_from_data_set(command, length)
    model.save(command)
    print "Model built and saved to " + model.get_model_data_path(command)
    while True:
        command = raw_input("Type words to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print "model[{0}] = {1}".format(command, str(model[command]))
        except KeyError:
            print "No such word in model"
