"""
Contains class representing Word2Vec embedding, implementing IWordEmbedding interface
"""
import os
import itertools
import multiprocessing
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from src.data import make_dataset
from iword_embedding import IWordEmbedding, TextCorpora


class Word2VecEmbedding(IWordEmbedding):
    visualization_sample_size = 100

    def __init__(self, text_corpus):
        self.text_corpus = text_corpus
        self.model = {}
        # self.preprocess = lambda x: x
        # self.pca = PCA(n_components=IWordEmbedding.target_vector_length)

    def saved_embedding_exists(self, data_folder):
        embedding_file_path = self.get_embedding_model_path(data_folder)
        return os.path.isfile(embedding_file_path)

    def save(self, output_path):
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        self.model.save(output_path)

    def load(self, data_path, sentences):
        self.model = Word2Vec.load(data_path)
        # self.build_preprocess_transformation(sentences)

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
        vec_length = IWordEmbedding.initial_vector_length
        total_corpus = itertools.chain(self.text_corpus, sentences)
        cpu_count = multiprocessing.cpu_count()
        self.model = Word2Vec(total_corpus, size=vec_length, min_count=1, workers=cpu_count)
        self.model.init_sims(replace=True)  # finalize the model
        # self.build_preprocess_transformation(sentences) uncomment to fit PCA on single words

    def __getitem__(self, word):
        if word not in self.model or word == '':
            return [0.0] * IWordEmbedding.target_vector_length
        return self.model[[word]][0]
        # return self.preprocess(self.model[[word]])[0]

    def get_embedding_model_path(self, data_folder):
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
