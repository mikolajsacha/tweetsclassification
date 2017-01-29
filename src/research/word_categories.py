""" I will try to train classifier to recognize category of single word.
    I will try N classifiers, where N = number of categories.
    Each classifier will give a number which should indicate, how much given word is associated with a category """

from sklearn.ensemble import RandomForestRegressor
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.common import LABELS, SENTENCES, CATEGORIES, CATEGORIES_COUNT, DATA_FOLDER
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
import numpy as np


def get_words_scores(labels, sentences):
    """ Returns list of tuples (word, [scores in categories])"""
    words_counts = {}
    for label, sentence in zip(labels, sentences):
        for word in (w for w in sentence if not w.isspace()):
            if word not in words_counts:
                words_counts[word] = [0.0] * CATEGORIES_COUNT
            words_counts[word][label] += 1.0

    for word, counts in words_counts.iteritems():
        minimum_count = min(counts)
        for i in xrange(CATEGORIES_COUNT):
            counts[i] -= minimum_count

    for i in xrange(CATEGORIES_COUNT):
        category_max = max(counts[i] for word, counts in words_counts.iteritems())
        for counts in words_counts.itervalues():
            counts[i] /= category_max
            counts[i] *= 100
    return words_counts


def get_words_categories_regressors(labels, sentences, word_emb, verbose=False):
    # count how often words occur in each category and make as score out of it
    words_scores = get_words_scores(labels, sentences)

    if verbose:
        print ("5 words with best scores for each category:")
        for i in xrange(CATEGORIES_COUNT):
            print CATEGORIES[i] + ": " + ', '.join("{:s}: {:4.2f}".format(w, v[i]) for w, v in
                                                   sorted(words_scores.iteritems(), key=lambda (x, y): -y[i])[:10])

    if verbose:
        print ("\nBuilding word embedding...")

    word_vectors = [word_emb[word] for word in words_scores.iterkeys()]

    if verbose:
        print ("\nEvaluating models predicting probabilities of each category for words...")

    categories_classifiers = []

    for i in xrange(CATEGORIES_COUNT):
        if verbose:
            print ("Evaluating model for category " + CATEGORIES[i] + "...")
        clf = RandomForestRegressor(n_estimators=100)
        values = np.array([v[i] for v in words_scores.itervalues()])

        # use values also as weights
        weights = values

        clf.fit(word_vectors, values, weights)
        categories_classifiers.append(clf)

    return categories_classifiers


if __name__ == "__main__":
    word_emb = Word2VecEmbedding(TextCorpora.get_corpus("brown"))
    word_emb.build(SENTENCES)

    classifiers = get_words_categories_regressors(LABELS, SENTENCES, word_emb, True)
    print ("Models evaluated!...\n")

    while True:
        command = raw_input("Type word to test its category or 'quit' to exit: ")
        if command.lower() == "quit" or command.lower() == "exit":
            break

        if command not in word_emb.model:
            print("No such word in word embedding")
            continue

        similar_words = word_emb.model.most_similar(command)
        print("Similar words:")
        print ', '.join("{:s}: {:0.4f}".format(w, sim * 100) for w, sim in similar_words)

        print("Predicted scores:")
        predictions = [clf.predict([word_emb[command]])[0] for clf in classifiers]
        print "\n".join("{:s}: {:4.4f}".format(CATEGORIES[i], prob) for i, prob in enumerate(predictions))

