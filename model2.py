# -*- coding: utf-8 -*-
"""


@author: Anaji
"""
# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# load the model from disk

df= pd.read_csv("data/spam.csv", encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
# Features and Labels
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data

filename1 = 'model/tranform.pkl'
pickle.dump(cv, open(filename1, 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
filename2 = 'model/nlp_model.pkl'
pickle.dump(clf, open(filename2, 'wb'))


clf = pickle.load(open(filename2,'rb'))
cv=pickle.load(open(filename1,'rb'))

message = "You won 1000"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)
message = "Hello how are you"
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)