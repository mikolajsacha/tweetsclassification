import os

from src.common import CATEGORIES
from src.data.data_gathering.tweet_miner import get_mined_tweets_path


def get_selected_tweets_path():
    path = os.path.join(os.path.dirname(__file__), '../../../data/gathered_dataset/tweets_mining/selected_tweets.txt')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    return path

if __name__ == "__main__":
    categories = CATEGORIES
    mined_tweets_path = get_mined_tweets_path()
    selected_tweets_path = get_selected_tweets_path()

    print "Numbers of categories: " + \
          ",  ".join(map(lambda (i, v): "{0}: {1}".format(i, v),
                         enumerate(categories)))
    print "*" * 20

    counter = 0
    for tweet in open(mined_tweets_path, 'r'):
        tweet = tweet.strip().replace('\n', '')
        print tweet
        category = raw_input("Type category or its number (any other character == ignore this tweet): ")
        if category == 'exit' or category == 'quit':
            break
        try:
            if category in categories:
                cat_int = categories.index(category)
            else:
                cat_int = int(category)
            if 0 <= cat_int < len(categories):
                with open(selected_tweets_path, 'a') as output_file:
                    output_file.write("{0} {1}\n".format(cat_int, tweet))
                print "Tweet saved!"
            print "Numbers of categories: " + \
                  ",  ".join(map(lambda (i, v): "{0}: {1}".format(i, v),
                                 enumerate(categories)))
        except ValueError:
            pass
        finally:
            print "*" * 20
