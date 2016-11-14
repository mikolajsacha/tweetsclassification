"""
Contains class representing Word2Vec embedding, implementing IWordEmbedding interface
"""
from iword_embedding import IWordEmbedding
from gensim.models import Word2Vec


class Word2VecEmbedding(IWordEmbedding):
    def __init__(self):
        # for start build an empty model
        self.model = Word2Vec([], size=0)

    def load(self, file_path):
        self.model = Word2Vec.load(file_path)
        self.model.init_sims(replace=True)  # trim unneeded model memory

    def safe(self, file_path):
        self.model.save(file_path)

    def build(self, sentences, vector_length):
        self.model = Word2Vec(sentences, size=vector_length)
        self.model.init_sims(replace=True)  # trim unneeded model memory

    def __getitem__(self, word):
        return self.model[word]

    def get_model_data_path(self):
        return IWordEmbedding.get_model_data_path() + '/Word2Vec.txt'

