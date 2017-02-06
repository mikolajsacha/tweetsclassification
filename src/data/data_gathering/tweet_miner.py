import Queue
import multiprocessing
from threading import Thread

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import unicodedata
import json
import os

from src.common import LABELS, SENTENCES
from src.features import build_features
from src.models.algorithms.nearest_neighbors_algorithm import NearestNeighborsAlgorithm
from src.models.algorithms.neural_network import NeuralNetworkAlgorithm
from src.models.algorithms.random_forest_algorithm import RandomForestAlgorithm
from src.models.algorithms.svm_algorithm import SvmAlgorithm
from src.features.word_embeddings.word2vec_embedding import Word2VecEmbedding
from src.features.word_embeddings.glove_embedding import GloveEmbedding
from src.features.sentence_embeddings.sentence_embeddings import *
from src.models.model_testing.grid_search import get_best_from_grid_search_results

access_token = "762272040120356864-GwHxKFisY8vBPsXhZ9YHo9DHSPFbpN8"
access_token_secret = "KyCIwI5aitMJGj07y7W361VYU5EBGEGl9luircvfdfpyb"
consumer_key = "91TJHUbpaRsGZ3cAXzjPs0Uaw"
consumer_secret = "pkdaCqfr4Qve8ewqYopYSv8PHdYDo98Sue9JmHNuxIq0RORppd"

tweets_queue = Queue.Queue()
clf = None
lock = multiprocessing.Lock()


def get_mined_tweets_path():
    path = os.path.join(os.path.dirname(__file__), '../../../data/gathered_dataset/tweets_mining/mined_tweets.txt')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path


class FileListener(StreamListener):
    tweets_file_path = get_mined_tweets_path()
    def __init__(self, ):
        super(FileListener, self).__init__()

    def on_data(self, data):
        try:
            tweet = json.loads(data)
            if "lang" in tweet and tweet["lang"] == "en":
                text = unicodedata.normalize('NFKD', tweet["text"]).encode('ascii', 'ignore')
                text = text.strip().replace('\n', '')
                tweet_words = text.split()
                if len(tweet_words) >= 5:
                    tweets_queue.put(text)
        except Exception as e:
            print "Exception: " + str(e)
        finally:
            return True

    def on_error(self, status):
        print status


def tweets_filter(classifier_threshold, include_unclassified):
    tweets_file_path = get_mined_tweets_path()
    while True:
        try:
            tweet = tweets_queue.get(timeout=10)
        except Queue.Empty:
            print ("Queue was empty for 10 seconds")
            break
        tweets_queue.task_done()
        probas = clf.predict_proba(tweet)
        if (include_unclassified and any(x >= classifier_threshold for x in probas)) or \
           (not include_unclassified and any(x >= classifier_threshold for x in probas[:-1])):
            with lock:
                print tweet, ', '.join(["{:4.4f}".format(p) for p in probas])
                with open(tweets_file_path, 'a')  as f:
                   f.write(tweet + '\n')


def mine_tweets(classifier_threshold, include_unclassified):
    global clf

    # building estimator with possibly best performance on already gathered tweets
    clf_class, word_emb_class, word_emb_params, sen_emb_class, params = get_best_from_grid_search_results()

    print ("\nEvaluating model for {:s}, word embedding: {:s}({:s}), sentence embedding: {:s} \nHyperparameters {:s}\n"
           .format(clf_class.__name__, word_emb_class.__name__,
                   ', '.join(map(str, word_emb_params)), sen_emb_class.__name__, str(params)))
    params["n_jobs"] = -1 # use multi-threading

    print ("Building word embedding...")
    word_emb = word_emb_class(*word_emb_params)
    word_emb.build()
    sen_emb = sen_emb_class()
    sen_emb.build(word_emb)

    print ("Building features...")
    fb = build_features.FeatureBuilder()
    fb.build(sen_emb, LABELS, SENTENCES)

    print ("Building model...")
    clf = clf_class(sen_emb, probability=True, **params)
    clf.fit(fb.features, fb.labels)

    print ("Model evaluated!...")

    l = FileListener()

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    threads = [Thread(target=tweets_filter, args=(classifier_threshold, include_unclassified))
               for _ in xrange(max(1, multiprocessing.cpu_count() - 1))]
    for t in threads:
        t.start()

    print ("Waiting for incoming tweets...\n")
    while True:
        try:
            stream.filter(locations=[-180, -90, 180, 90])
        except Exception:
            print("Re-establishing lost connection")


if __name__ == "__main__":
    mine_tweets(0.7, False)