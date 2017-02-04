"""
Contains class representing Word2Vec embedding, implementing IWordEmbedding interface
"""
import warnings
import itertools
from src.features.word_embeddings.iword_embedding import IWordEmbedding

#  This import generates an annoying warning on Windows
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from gensim.models import Word2Vec


class Word2VecEmbedding(IWordEmbedding):
    def __init__(self, path, vector_length):
        IWordEmbedding.__init__(self, path, vector_length)

    def _build(self):
        self.model = Word2Vec.load_word2vec_format(self.get_embedding_model_path(), binary=True)

    def __getitem__(self, word):
        if word not in self.model or word.isspace():
            return None
        return self.model[[word]][0]

if __name__ == "__main__":
    """
    Main method allows to interactively build and test Word2Vec model
    """

    model = Word2VecEmbedding('google/GoogleNews-vectors-negative300.bin', 300)
    model.build()
    while True:
        command = raw_input("Type words to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print "most similar words: "
            words = command.split(' ')
            if len(words) == 1:
                similar_words = model.model.most_similar(words[0])
                print u',  '.join(u"{:s} : {:4.4f}".format(w, sim*100) for w, sim in similar_words)
            else:
                for w1, w2 in (p for p in itertools.product(words, words) if p[0] < p[1]):
                    print "Similarity of {0} and {1}: {2}".format(w1, w2, model.model.similarity(w1, w2))
        except KeyError:
            print "No such word in model"
