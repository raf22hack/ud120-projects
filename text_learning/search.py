#!/usr/bin/python

import os
import pickle
import re
import sys

document = open("your_word_data_fixed.pkl", "rb")
pickle_obj = pickle.load(document)


print(len(pickle_obj))


###########################


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
vectorized_corpus = vectorizer.fit_transform(pickle_obj)

print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[33614])