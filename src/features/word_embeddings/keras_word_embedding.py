"""
Contains class with own word embedding build using embedding layer from keras library
"""
import os
import itertools
from keras.layers import Embedding
from keras.models import Sequential
from src.data import make_dataset
from iword_embedding import IWordEmbedding, TextCorpora


class KerasWordEmbedding(IWordEmbedding):
    def __init__(self, text_corpus, vector_length=40):
        IWordEmbedding.__init__(self, text_corpus, vector_length)
        self.model = None
        self.indices = {}

    def build(self, sentences):
        total_corpus = list(itertools.chain(self.text_corpus, sentences))
        curr_index = 0
        for sentence in total_corpus:
            for word in sentence:
                if word not in self.indices:
                    self.indices[word] = curr_index
                    curr_index += 1

        numbered_corpus = [[self.indices[word] for word in sentence] for sentence in total_corpus]

        model = Sequential()
        model.add(Embedding(curr_index, self.vector_length))
        model.compile('rmsprop', 'mse')

        self.model = model.predict(numbered_corpus)

    def __getitem__(self, word):
        if word not in self.indices or word == '':
            return [0.0] * self.vector_length
        return self.model[self.indices[word]] # TODO: this does not work (could not test on windows)

    @staticmethod
    def get_embedding_model_path(data_folder):
        return IWordEmbedding.get_embedding_model_path(data_folder) + '\\keras'


if __name__ == "__main__":
    """
    Main method allows to interactively build and test model
    """

    while True:
        command = raw_input("Type data set folder name to build Word2Vec embedding: ")
        input_file_path = make_dataset.get_processed_data_path(command)

        if not os.path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)
        else:
            break

    data_info = make_dataset.read_data_info(make_dataset.get_data_set_info_path(command))
    labels, sentences = make_dataset.read_dataset(input_file_path, data_info)

    print("Building embedding...")
    model = KerasWordEmbedding(TextCorpora.get_corpus("brown"))
    model.build(sentences)
    print("Saving model to a file...")
    model.save(model.get_embedding_model_path(command))
    print "Model built and saved to " + model.get_embedding_model_path(command)
    while True:
        command = raw_input("Type words to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print "model[{0}] = {1}".format(command, str(model[command]))
        except KeyError:
            print "No such word in model"
