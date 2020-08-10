import re
import string

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def process_text(message):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers 
    message = re.sub(r'\$\w*', '', message)
    # remove hyperlinks
    message = re.sub(r'https?:\/\/.*[\r\n]*', '', message)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    message_tokens = tokenizer.tokenize(message)

    message_clean = []
    for word in message_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):   # remove punctuation
            stem_word = stemmer.stem(word)     # stemming word
            message_clean.append(stem_word)

    return message_clean
