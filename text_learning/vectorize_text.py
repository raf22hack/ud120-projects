#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset

        ### Temp Counter Header
        # temp_counter += 1
        # if temp_counter < 200:

        path = os.path.join('..', path[:-1])
        print(path)
        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email

        processed_email = parseOutText(email)

        print(processed_email)

        ### use str.replace() to remove any instances of the words
        ### ["sara", "shackleton", "chris", "germani"]

        replace_list = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]#, "houectect", "houect", "houston"] #, "sshacklmsncom"]

        replaced_email = processed_email

        for name in replace_list:
            replaced_email = replaced_email.replace(name, '')

        ### append the text to word_data

        word_data.append(replaced_email)

        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris

        if from_person == from_chris:
            from_data.append(1)
        elif from_person == from_sara:
            from_data.append(0)


        email.close()

print(from_data)

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data_fixed.pkl", "wb") )
pickle.dump( from_data, open("your_email_authors_fixed.pkl", "wb") )

### in Part 4, do TfIdf vectorization here

### IF YOU USE PICKLE LOAD YOU COULD SAVE TONS OF TIME!

### PICKLE LOAD ! PICKLE LOAD !

### PICKLE LOAD ! PICKLE LOAD !

### PICKLE LOAD ! PICKLE LOAD !


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english")
vectorized_corpus = vectorizer.fit_transform(word_data)

print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[34597])

# # print(vectorized_corpus)
# # print(vectorized_corpus.shape)
# print("Type of the matrix:", type(vectorized_corpus))
# print("Shape of the matrix:", vectorized_corpus.get_shape())
# print("E-mails transferred into matrix:", len(word_data))
#
value = vectorized_corpus.data
column_index = vectorized_corpus.indices
row_pointers = vectorized_corpus.indptr
#
# print("------------------")
#
# print(vectorized_corpus)
#
# print("------------------")
# print("VALUES?")
# print("Begin Values:\n", value[:10], "\nTotal lenght of...", len(value))
# print("End Values:\n", value[10:])
#
# print("------------------")
# print("COLUMN INDEX")
# print("Begin Column Index:", column_index[:10], "\nTotal lenght of...", len(column_index))
# print("End Column Index:", column_index[10:])
#
# print("------------------")
# print("ROW POINTERS")
# print("Begin Row Pointers:", row_pointers[:10], "\nTotal lenght of...", len(row_pointers))
# print("End Row Pointers:", row_pointers[10:])
#
# # print(vectorized_corpus[198, 2858])
# #
# # word_index = 1
# #
# # for i in row_pointers:
# #     try:
# #         print("Index:", i, "Value:", vectorized_corpus[i - 1, word_index])
# #     except:
# #         continue
#
# print("------------------")
#
# word_index = 34597
#
# for i in row_pointers:
#     try:
#         if vectorized_corpus[i - 1, word_index] != 0:
#             print("Row:", i, "Column:", word_index, "Value:", vectorized_corpus[i - 1, word_index])
#             print(word_data[i])
#     except:
#         continue
#
# for i in row_pointers:
#     for j in column_index:
#         try:
#             if j == word_index:
#                 print("Row:", i, "Column:", j, "Value:", vectorized_corpus[i - 1, j])
#                 print(word_data[i])
#         except:
#             continue