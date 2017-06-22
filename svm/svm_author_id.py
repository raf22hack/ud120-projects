#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# clf = SVC(kernel="linear")

# c_parameters = [10.0, 100.0, 1000.0, 10000.0]

i = 10000.0

clf = SVC(kernel="rbf", C=i)

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train, labels_train)
print("Training time (C=", i, "): ", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("Prediction time (C=", i, "): ", round(time()-t1, 3), "s")

# print(pred[10])
# print(pred[26])
# print(pred[50])

print(len(pred))

chris = []
sara = []

for i in pred:
    if i == 1:
        chris.append(i)
    else:
        sara.append(i)

print(len(chris))
print(sum(chris))

acc = accuracy_score(pred, labels_test)

# def submitAccuracy():
#     return acc

print("Accuracy (C=", i, "): ", acc)

    #########################################################
### your code goes here ###

#########################################################


