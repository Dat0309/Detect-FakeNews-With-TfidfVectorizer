import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#read dataset
data = pd.read_csv('news.csv')
data.shape
data.head()
print(data.head())

#get labels
labels = data.label
labels.head()
print(labels.head())

#Split the dataset
trainX, trainY, testX, testY = train_test_split(data['text'],labels, test_size=0.2, random_state=7)

'''
 Let’s initialize a TfidfVectorizer with stop words from the English language and a maximum document
frequency of 0.7 (terms with a higher document frequency will be discarded). Stop words are the most 
common words in a language that are to be filtered out before processing the natural language data. 
And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
'''
#Initialize a TFIDFVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#fit and transform 
tfidf_train = tfidf_vectorizer.fit_transform(trainX)
tfidf_test = tfidf_vectorizer.transform(testX)

#initialize a passiveAggressiveClassifer
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, trainY)

#predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(testY, y_pred)
print(f'Accuracy: (round(score*100,2))%')

#Build confusion matrix
confusion_matrix(testY, y_pred, labels=['FAKE','REAL'])