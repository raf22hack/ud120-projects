#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from operator import itemgetter


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_fixed.pkl", "rb") )

x = 0

for key in data_dict:
    x += 1
    if key == "TOTAL":
        print("FOUND IT!!!!")
        data_dict.pop("TOTAL", None)
        break
    # print(key)


features = ["bonus", "salary"]
data = featureFormat(data_dict, features)

keys = []
temp_list = []

salary_var = 0
bonus_var = 0

for key in data_dict:

    if data_dict[key]["salary"] == "NaN":
        salary_var = 0
    else:
        salary_var = int(data_dict[key]["salary"])

    if data_dict[key]["bonus"] == "NaN":
        bonus_var = 0
    else:
        bonus_var = int(data_dict[key]["bonus"])

    temp_list = key, salary_var, bonus_var
    keys.append(temp_list)

sorted_keys = sorted(keys, key=itemgetter(2), reverse=True)

print(sorted_keys)

### your code below

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

bonus, salary = targetFeatureSplit(data)

# print(len(salary))
#
# for i in salary:
#     print(i.tolist()[0])

salary_train, salary_test, bonus_train, bonus_test = train_test_split(salary, bonus, test_size=0.1, random_state=42)

reg = LinearRegression()
reg.fit(salary_train, bonus_train)

plt.plot(salary_test, reg.predict(salary_test), color="blue")

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( bonus, salary )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()
