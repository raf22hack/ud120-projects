#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print("Feature Lenght: ", len(features_train[0]))

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split=40)

t1 = time()
clf.fit(features_train, labels_train)
print("Training time: ", round(time()-t1, 3), "S")

t2 = time()
pred = clf.predict(features_test)
print("Prediction time: ", round(time()-t2, 3), "s")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)

print(accuracy)


#########################################################
### your code goes here ###


#########################################################


