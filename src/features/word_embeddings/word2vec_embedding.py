"""
Contains class representing Word2Vec embedding, implementing IWordEmbedding interface
"""
from os import path, makedirs

import itertools
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from iword_embedding import IWordEmbedding
from src.data.make_dataset import get_processed_data_path
from nltk.corpus import brown


class Word2VecEmbedding(IWordEmbedding):
    visualization_sample_size = 100

    def __init__(self):
        self.model = {}
        self.preprocess = lambda x: x
        self.pca = PCA(n_components=IWordEmbedding.target_vector_length)

    def saved_embedding_exists(self, data_folder):
        embedding_file_path = self.get_embedding_model_path(data_folder)
        return path.isfile(embedding_file_path)

    def save(self, data_folder):
        output_path = self.get_embedding_model_path(data_folder)
        if not path.exists(path.dirname(output_path)):
            makedirs(path.dirname(output_path))
        self.model.save(output_path)

    def load(self, data_folder, sentences):
        self.model = Word2Vec.load(self.get_embedding_model_path(data_folder))
        self.build_preprocess_transformation(sentences)

    def build_preprocess_transformation(self, training_set_sentences):
        # preprocessing is built only by words from our training set
        training_set_vectors = []
        already_in_training_set = set()
        for sentence in training_set_sentences:
            for word in sentence:
                if word not in already_in_training_set:
                    already_in_training_set.add(word)
                    training_set_vectors.append(self.model[word])
        self.pca.fit(training_set_vectors)
        self.preprocess = lambda vector: self.pca.transform(vector)

    def build(self, sentences):
        text_corpus = iter(brown.sents())
        vector_length = IWordEmbedding.initial_vector_length
        self.model = Word2Vec(itertools.chain(text_corpus, sentences), size=vector_length, min_count=1)
        self.model.init_sims(replace=True)  # finalize the model
        self.build_preprocess_transformation(sentences)

    def __getitem__(self, word):
        if word in self.model:
            return self.preprocess(self.model[[word]])[0]
        return self.preprocess(self.model[['']])[0]

    def get_embedding_model_path(self, data_folder):
        return IWordEmbedding.get_embedding_model_path(data_folder) + '\\word2vec'


if __name__ == "__main__":
    """
    Main method allows to interactively build and test Word2Vec model
    """

    while True:
        command = raw_input("Type data set folder name to build Word2Vec embedding: ")
        input_file_path = get_processed_data_path(command)

        if not path.isfile(input_file_path):
            print "Path {0} does not exist".format(input_file_path)
        else:
            break

    print("Building model...")
    model = Word2VecEmbedding()
    model.build_from_data_set(command)
    print("Saving model to a file...")
    model.save(command)
    print "Model built and saved to " + model.get_embedding_model_path(command)
    while True:
        command = raw_input("Type words to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print "model[{0}] = {1}".format(command, str(model[command]))
        except KeyError:
            print "No such word in model"
