import json
import os
import pickle
import random

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem.lancaster import LancasterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
stemmer = LancasterStemmer()
words = []
classes = []
documents = []
ignore_words = ['?']

bFile = "src/b-data.pkl"
bmod = "src/b-model.pkl"

def check_files():
    """
    Creates empty encrypted files 
    """
    files = { 
    }

    for filename, value in files.items():
        if not os.path.isfile("{}".format(filename)):
            print("Creating empty {}".format(filename))
            with open("{}".format(filename), 'wb') as outfile:
                json.dump(value, outfile)
                outfile.close()


class outfitModule():
    """
    in the class it will:
        loop through each sentence in the intents patterns
        Create model with 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        equal to number of intents to predict output intent with softmax
        Output is a '0' for each tag and '1' for current tag (for each pattern)
    """
    def main():
        global words
        global classes
        global documents
        global ignore_words
        with open("src/data.json", 'rb') as mldbb:
            intents = json.load(mldbb)

        for intent in intents['intents']:
            for pattern in intent['patterns']:
                try:
                    w = nltk.word_tokenize(pattern)
                except:
                    nltk.download()  # This is done to add packages
                words.extend(w)
                documents.append((w, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        print(len(documents), "documents")
        print(len(classes), "classes", classes)
        print(len(words), "unique stemmed words", words)  # all words, vocabulary

        # create our training data
        training = []
        output_empty = [0] * len(classes)
        for doc in documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [stemmer.stem(word.lower())
                             for word in pattern_words]
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        RinData = {"words": words, "classes": classes}
        modelPickleData = open(bFile, 'wb')
        pickle.dump(RinData, modelPickleData)
        modelPickleData.close()

        modelPickle = open(bmod, 'wb')
        pickle.dump(model, modelPickle)
        modelPickle.close()


class rinProcess():
    def __init__(self):
        check_files()
        self.model = pickle.load(open(bmod, 'rb'))
        self.data = pickle.load(open(bFile, "rb"))
        self.words = self.data['words']
        self.classes = self.data['classes']
    """
    in this class it will:
        Use pickle to load in the pre-trained model then generate probabilities from the model.
        After it will filter out predictions below a threshold, and provide intent index then sort by strength of probability.
        and values will return tuple of intent and probability
    """

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words


    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return(np.array(bag))


    def think(self, sentence):
        ERROR_THRESHOLD = 0.85

        input_data = pd.DataFrame([self.bow(sentence, self.words)], dtype=float, index=['input'])
        results = self.model.predict([input_data])[0]
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=False)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], str(r[1])))
            
        return return_list


def chat():
    rinProcess().think("Hello!")
    print("Start talking with the bot (type quit to stop)!")
    while True:
        try:
            inp = input("<USER>: ")
            if inp.lower() == "quit":
                break
            if inp.lower() == "rinfit":
                d = outfitModule
                d.main()
            p = rinProcess().think(inp)
            print(p)
            with open('src/data.json', "rb") as json_data:
                intents = json.load(json_data)

            for x in range(0, len(intents["intents"])):
                if p == [] or str(p[0][0]) == "untrained":
                    result = {'pos': 0, 'neg': 0, 'neu': 0}
                    score = analyzer.polarity_scores(inp)
                    if score['compound'] > 0.05:
                        result['pos'] += 1
                    elif score['compound'] < -0.05:
                        result['neg'] += 1
                    else:
                        result['neu'] += 1

                    intentsList = [result]
                elif str(intents["intents"][x]["tag"]) == str(p[0][0]):
                    intentsList = intents["intents"][x]["responses"]

            # print(p)
            print("<Oybot>:", random.choice(intentsList))

        except KeyboardInterrupt:
            break

chat()