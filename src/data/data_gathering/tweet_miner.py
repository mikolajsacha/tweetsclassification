import ast
import os

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import unicodedata
import json

from src.data import dataset
from src.features import build_features
from src.features.word_embeddings.iword_embedding import TextCorpora
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.models.model_testing.grid_search import get_grid_search_results_path
from src.features.word_embeddings.word2vec_embedding import *
from src.features.sentence_embeddings.sentence_embeddings import *

access_token = "762272040120356864-GwHxKFisY8vBPsXhZ9YHo9DHSPFbpN8"
access_token_secret = "KyCIwI5aitMJGj07y7W361VYU5EBGEGl9luircvfdfpyb"
consumer_key = "91TJHUbpaRsGZ3cAXzjPs0Uaw"
consumer_secret = "pkdaCqfr4Qve8ewqYopYSv8PHdYDo98Sue9JmHNuxIq0RORppd"
categories_keywords = []
categories = []
all_keywords = set()


def fits_keyword(word, keyword):
    return (keyword == word) or (len(keyword) > 5 and keyword in word and len(word) - len(keyword) < 3)


class FileListener(StreamListener):
    classifier_threshold = 0.5

    def __init__(self, file_name, clf, categories):
        super(FileListener, self).__init__()
        self.file_name = file_name
        self.clf = clf
        self.categories = categories

    def is_classified(self, tweet_text):
        probas = self.clf.predict_proba(tweet_text)
        if filter(lambda x: x > FileListener.classifier_threshold, probas):
            return True
        return False

    def on_data(self, data):
        try:
            tweet = json.loads(data)
            if "lang" in tweet and tweet["lang"] == "en":
                text = unicodedata.normalize('NFKD', tweet["text"]).encode('ascii', 'ignore')
                text = text.strip().replace('\n', '')
                tweet_words = text.split()
                if len(tweet_words) < 4:
                    pass
                keyphrases_in_tweet = filter(lambda k: ' ' in k and k in tweet, tweet_words)
                keywords_in_tweet = filter(lambda w: len(filter(lambda k: fits_keyword(w, k), all_keywords)) > 0,
                                           tweet_words)
                if keyphrases_in_tweet or keywords_in_tweet:
                    with open(self.file_name, 'a') as tweets_file:
                        tweets_file.write(text + "\n")
                    print "Saved tweet: " + text + \
                          " {0}".format(map(lambda (i, prob):
                                            "{:s}: {:4.2f}%".format(self.categories[i], 100.0 * prob),
                                            enumerate(self.clf.predict_proba(text))))
        except Exception as e:
            print "Exception: " + str(e)
        finally:
            return True

    def on_error(self, status):
        print status


if __name__ == "__main__":
    #  building estimator with possibly best performance on already gathered tweets
    data_folder = "gathered_dataset"
    data_path = dataset.get_processed_data_path(data_folder)
    data_info = dataset.read_data_info(dataset.get_data_set_info_path(data_folder))
    labels, sentences = dataset.read(data_path, data_info)
    classifiers = [SvmAlgorithm, NeuralNetworkAlgorithm, RandomForestAlgorithm]

    max_result = 0.0
    best_parameters = None
    best_classifier = None

    for classifier in classifiers:
        summary_file_path = get_grid_search_results_path(data_folder, classifier)

        if not (os.path.exists(summary_file_path) and os.path.isfile(summary_file_path)):
            continue

        print("Found Grid Search results in " + summary_file_path.split("..")[-1])
        for line in open(summary_file_path, 'r'):
            embedding, params, result = tuple(line.split(";"))
            result = float(result)
            if result > max_result:
                max_result = result
                best_parameters = embedding, ast.literal_eval(params)
                best_classifier = classifier

    if best_classifier is None:
        print "No Grid Search results found for any estimator."
        exit(-1)

    embedding, params = best_parameters
    word_emb_class, sen_emb_class = tuple(embedding.split(","))

    print ("\nEvaluating model for classifier {:s} with embedding {:s} and params {:s} (performance: {:4.2f}%)\n"
           .format(best_classifier.__name__, embedding, str(params), max_result))

    print ("Building word embedding...")
    word_emb = eval(word_emb_class)(TextCorpora.get_corpus("brown"))
    word_emb.build(sentences)
    sen_emb = eval(sen_emb_class)()

    print ("Building sentence embedding...")
    sen_emb.build(word_emb, labels, sentences)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, labels, sentences)

    print ("Building model...")
    clf = best_classifier(sen_emb, probability=True, **params)
    clf.fit(fb.features, fb.labels)

    print ("Model evaluated!...")

    for line in open('keywords.txt', 'r'):
        category, keywords_str = tuple(line.split(':'))
        keywords = keywords_str.rstrip().split(',')
        categories.append(category)
        categories_keywords.append(keywords)
    all_keywords = set(item for sublist in categories_keywords for item in sublist)
    if '' in all_keywords:
        all_keywords.remove('')

    l = FileListener('tweets.txt', clf, data_info['Categories'])

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    print ("Waiting for incoming tweets...\n")
    stream.filter(locations=[-180, -90, 180, 90])
