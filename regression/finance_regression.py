#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from operator import itemgetter
dictionary = pickle.load( open("../final_project/fp_dataset_modified_fixed.pkl", "rb") )

enron_data = dictionary

keys = []
temp_list = []

salary_var = 0
bonus_var = 0

for key in enron_data:

    if enron_data[key]["salary"] == "NaN":
        salary_var = 0
    else:
        salary_var = int(enron_data[key]["salary"])

    if enron_data[key]["bonus"] == "NaN":
        bonus_var = 0
    else:
        bonus_var = int(enron_data[key]["bonus"])

    temp_list = key, salary_var, bonus_var
    keys.append(temp_list)

sorted_keys = sorted(keys, key=itemgetter(2), reverse=True)

print(sorted_keys)
# print(max(enron_data[])

# # This prints the content of each key, which is a dictionary
# for key in enron_data:
#     try:
#         if int(enron_data[key]["bonus"]) < 100000:
#             print(key, enron_data[key])
#     except:
#         continue

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )


### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score

reg = LinearRegression()
reg.fit(feature_train, target_train)


print(reg.coef_)
print(reg.score(feature_test,target_test))
print(np.mean((reg.predict(feature_test) - target_test) ** 2))
print(r2_score(target_test,reg.predict(feature_test)))




### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="g")

print(reg.coef_)
print(reg.intercept_)

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()


# import numpy as np
#
# pred = reg.predict(feature_test)
#
#
# pred_list = np.ndarray.tolist(pred)
#
# print(len(feature_test))
# print(target_test)
# print(pred_list)
# #
# # reg.score(pred_list, target_test)
#
# r2score = r2_score(reg.predict(feature_test), target_test)
#
# print("M =", reg.coef_, "I =" ,reg.intercept_)
# print("Score =", r2score)