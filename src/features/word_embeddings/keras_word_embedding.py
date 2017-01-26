"""
Contains class with own word embedding build using embedding layer from keras library
"""
import os
import itertools
from keras.layers import Embedding
from keras.models import Sequential
from src.data import make_dataset
from iword_embedding import IWordEmbedding, TextCorpora
import numpy as np


class KerasWordEmbedding(IWordEmbedding):
    def __init__(self, text_corpus, vector_length=40):
        IWordEmbedding.__init__(self, text_corpus, vector_length)
        self.model = {}

    def build(self, sentences):
        self.model = {}
        indices = {}
        total_corpus = list(itertools.chain(self.text_corpus, sentences))
        curr_index = 1
        for sentence in total_corpus:
            for word in sentence:
                if word not in indices:
                    indices[word] = curr_index
                    curr_index += 1

        max_sentence_length = max(map(len, total_corpus))
        numbered_corpus = np.zeros((len(total_corpus), max_sentence_length))
        for i, sentence in enumerate(total_corpus):
            for j, word in enumerate(sentence):
                numbered_corpus[i][j] = indices[word]

        model = Sequential()
        model.add(Embedding(curr_index, self.vector_length, input_length=max_sentence_length))
        model.compile('rmsprop', 'mse')

        model_matrix = model.predict(numbered_corpus)
        for i, sentence in enumerate(total_corpus):
            for j, word in enumerate(sentence):
                if word not in self.model:
                    self.model[word] = model_matrix[i][j]

    def __getitem__(self, word):
        if word not in self.model or len(word) <= 1:
            return [0.0] * self.vector_length
        return self.model[word]

    def get_nearest(self, word):
        if word not in self.model:
            raise KeyError("No such word in embedding")
        word_vec = self.model[word]
        min_dist = 1000
        min_word = None
        for other_word, other_word_vec in self.model.iteritems():
            if other_word == word:
                continue
            dist_square = sum((a - b) ** 2 for a, b in zip(word_vec, other_word_vec))
            if dist_square < min_dist:
                min_dist = dist_square
                min_word = other_word
        return min_word

    @staticmethod
    def get_embedding_model_path(data_folder):
        return IWordEmbedding.get_embedding_model_path(data_folder) + '\\keras'


if __name__ == "__main__":
    """
    Main method allows to interactively build and test model
    """

    while True:
        command = raw_input("Type data set folder name to build Keras embedding: ")
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
    while True:
        command = raw_input("Type words to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print "model[{0}] = {1}".format(command, str(model[command]))
            print "nearest word: {0}".format(model.get_nearest(command))
        except KeyError:
            print "No such word in model"
