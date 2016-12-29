"""
Contains class with own word embedding build using embedding layer from keras library
"""
import os
import warnings
import itertools
import multiprocessing
from src.data import make_dataset
from iword_embedding import IWordEmbedding, TextCorpora

#  This import generates an annoying warning on Windows
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from gensim.models import Word2Vec


class Word2VecEmbedding(IWordEmbedding):
    def __init__(self, text_corpus, vector_length=40):
        IWordEmbedding.__init__(self, text_corpus, vector_length)
        self.model = {}

    def saved_embedding_exists(self, data_folder):
        embedding_file_path = self.get_embedding_model_path(data_folder)
        return os.path.isfile(embedding_file_path)

    def save(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        self.model.save(output_path)

    def load(self, data_path, sentences):
        self.model = Word2Vec.load(data_path)

    def build(self, sentences):
        total_corpus = itertools.chain(self.text_corpus, sentences)
        cpu_count = multiprocessing.cpu_count()
        self.model = Word2Vec(total_corpus, size=self.vector_length, min_count=1, workers=cpu_count)
        self.model.init_sims(replace=True)  # finalize the model

    def __getitem__(self, word):
        if word not in self.model or word == '':
            return [0.0] * self.vector_length
        return self.model[[word]][0]

    @staticmethod
    def get_embedding_model_path(data_folder):
        return IWordEmbedding.get_embedding_model_path(data_folder) + '\\word2vec'


if __name__ == "__main__":
    """
    Main method allows to interactively build and test Word2Vec model
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
    model = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
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
