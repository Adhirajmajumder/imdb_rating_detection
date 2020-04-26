# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:49:48 2020

@author: ADHIRAJ MAJUMDAR
"""

import pandas as pd
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("./source data/imdb_labelled.txt",sep="  ",header=None)
data.columns = ["Text","Label"]

def message_parse(message):
    no_punctuation = [word for word in message if word not in punctuation]
    no_punctuation = ''.join(no_punctuation)
    return [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
X_train,X_test,Y_train,Y_test = train_test_split(data["Text"],data["Label"],test_size=0.25,random_state=11)
bag_of_words = CountVectorizer(analyzer=message_parse).fit(X_train)
bag_of_messages = bag_of_words.transform(X_train)

tfidf_words = TfidfTransformer().fit(bag_of_messages)
tfidf_message = tfidf_words.transform(bag_of_messages)
model = MultinomialNB().fit(tfidf_message,Y_train)

filename = './model/finalized_model_Naive_Bayes.sav'
pickle.dump(model, open(filename, 'wb'))

test_model = bag_of_words.transform(X_test)
test_model_transform = tfidf_words.transform(test_model)
prediction = model.predict(test_model_transform)
score = accuracy_score(prediction,Y_test)
X_test_sample = X_test[730]
sample_bag_of_words = bag_of_words.transform([X_test_sample])
sample_tfidf_transform = tfidf_words.transform(sample_bag_of_words)
print("Actual Message: ",X_test_sample)
print("Actual Label:",Y_test[730])
print("Predicted label: ",model.predict(sample_tfidf_transform)[0])
print("Model Accuracy:",score)