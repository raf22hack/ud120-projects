#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

dataset = open("test_dump.pkl", "rb")
enron_data = pickle.load(dataset)

first_key = next(iter(enron_data))

print(first_key)
print(enron_data[first_key])
# print(len(enron_data[first_key]))

# label = "poi"
# x = 0

# for i in enron_data:
#     current_key = next(iter(enron_data))
#     if current_key['poi'] == True:
#         x += 1

# for key in enron_data.items():
#     for value in key:
#         print(value[2])
#         # if value == label:
#         #     if value[] == True:
#         #         x += 1

# # This gives you only the keys.
# for key in enron_data:
#     print(key)

# # This prints the content of each key, which is a dictionary
# for key in enron_data:
#     print(key)
#     print(enron_data[key])

# # This loop prints all of the values in
# for key in enron_data:
#     for label in enron_data[key]:
#         print(enron_data[key][label])

# print(enron_data["PRENTICE JAMES"]["total_stock_value"])
# print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

# top = ["SKILLING JEFFREY K", "FASTOW ANDREW S", "LAY KENNETH L"]
# total = "total_payments"
#
# for i in top:
#     print(i, "=", enron_data[i][total])

poi = "poi"
label = "total_payments"
x = 0
y = 0
z = 0

for key in enron_data:
    if enron_data[key][poi] == True:
        if enron_data[key][label] == "NaN":
            print(key, ":", enron_data[key][label])
            x += 1
        z += 1
    y += 1

print()
print(x/len(enron_data), "%")
print(y)
print(z)
print(x)
