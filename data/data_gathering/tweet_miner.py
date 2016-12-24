from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import unicodedata
import json

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
    def __init__(self, file_name):
        super(FileListener, self).__init__()
        self.file_name = file_name

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
                    print "Saved tweet: " + text
        except Exception as e:
            print "Exception: " + str(e)
        finally:
            return True

    def on_error(self, status):
        print status


if __name__ == "__main__":
    for line in open('keywords.txt', 'r'):
        category, keywords_str = tuple(line.split(':'))
        keywords = keywords_str.rstrip().split(',')
        categories.append(category)
        categories_keywords.append(keywords)
    all_keywords = set(item for sublist in categories_keywords for item in sublist)
    all_keywords.remove('')

    l = FileListener('tweets.txt')

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    stream.filter(locations=[-180, -90, 180, 90])
