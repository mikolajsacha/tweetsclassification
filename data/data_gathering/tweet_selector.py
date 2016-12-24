
categories_keywords = []
categories = []
all_keywords = set()

if __name__ == "__main__":
    for line in open('keywords.txt', 'r'):
        category, keywords_str = tuple(line.split(':'))
        keywords = keywords_str.rstrip().split(',')
        categories.append(category)
        categories_keywords.append(keywords)
    all_keywords = set(item for sublist in categories_keywords for item in sublist)

    print "Numbers of categories: " + \
          ",  ".join(map(lambda (i, v): "{0}: {1}".format(i, v),
                         enumerate(categories)))
    print ""

    counter = 0
    with open('training_set.txt', 'a') as output_file:
        for tweet in open('tweets_to_mine.txt', 'r'):
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
                    output_file.write("{0} {1}\n".format(cat_int, tweet))
                    print "Tweet saved!"
                print "Numbers of categories: " + \
                      ",  ".join(map(lambda (i, v): "{0}: {1}".format(i, v),
                                     enumerate(categories)))
            except ValueError:
                pass
            finally:
                print ""

