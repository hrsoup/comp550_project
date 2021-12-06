import random
import os
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import json
import string
from copy import deepcopy
from nltk.tokenize import sent_tokenize

def stemming_preprocessor(text):
    # Initialize stemmer
    stemmer = PorterStemmer()

    # stem words
    stemmed_output = [stemmer.stem(word = w) for w in text]
    return stemmed_output

# function used to preprocess with lemmatizing
def lemmatize_preprocessor(text):
    # Initialize the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize list of words and join
    lemmatized_output = [lemmatizer.lemmatize(w) for w in text]
    return lemmatized_output

def preprocess_language():

    path = './language_dataset'
    files = os.listdir(path)
    X = []
    X_shuffled = []
    y = []
    for file in files:
        # Opening JSON file in same directory as python file
        f = open(path+'/'+file)
    
        # returns JSON object as
        # a dictionary
        data = json.load(f)
    
        pattern = re.compile('[^\t\n]+')

        # Iterating through the list of jokes
        for k in range(len(data)):
            sentences_shuffle = []
            sentences = []
            labels = []
            joke = data[k]['body']
            for sentence in sent_tokenize(joke.replace('\n','')):
                # remove all punctuation
                punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
                sentence = punctuation_regex.sub(' ',sentence)
                sentence = [val for values in map(pattern.findall, sentence.lower().split(' ')) for val in values]
                if len(sentence) >=2 :
                    label = [0] * len(sentence)
                    label[0] = 1
                    
                    # lematization on the joke
                    sentence = lemmatize_preprocessor(sentence)
                    # stemming on the joke
                    sentence = stemming_preprocessor(sentence)

                    sentence_shuffle = deepcopy(sentence)
                    random.shuffle(sentence_shuffle)

                    sentences_shuffle.extend(sentence_shuffle)
                    sentences.extend(sentence)
                    labels.extend(label)

            if len(sentences) < 128:
                if len(sentences) == len(labels):
                    X.append(sentences)
                    X_shuffled.append(sentences_shuffle)
                    y.append(labels)
        
        # Closing file
        f.close()

    return X, X_shuffled, y

X_language, X_shuffled_language, Y_language = preprocess_language()

X_unique = []
for i in range(len(X_language)):
    for item in X_language[i]:
        if (item in X_unique) == False:
            X_unique.append(item)

number_list = []
for i in range(len(X_unique)):
    number_list.append(i)
X_unique.sort()

word2id = dict(zip(X_unique, number_list))