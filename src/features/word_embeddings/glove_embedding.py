"""
Contains class representing downloaded pre-trained word embedding
"""
import itertools
from src.features.word_embeddings.iword_embedding import IWordEmbedding


class GloveEmbedding(IWordEmbedding):
    def __init__(self, path, vector_length):
        IWordEmbedding.__init__(self, path, vector_length)

    def _build(self):
        self.model = {}
        for line in open(self.get_embedding_model_path()):
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            self.model[word] = embedding
    def __getitem__(self, word):
        if word not in self.model or word.isspace():
            return None
        return self.model[word]

if __name__ == "__main__":
    """
    Main method allows to interactively build and test Glove model
    """
    model = GloveEmbedding('glove_twitter/glove.twitter.27B.200d.txt', 200)
    model.build()
    print "Model loaded"
    while True:
        command = raw_input("Type a word to test embedding or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break
        try:
            print model[command]
        except KeyError:
            print "No such word in model"
