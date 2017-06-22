#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    from operator import itemgetter
    cleaned_data = []
    temp_list = []

    ### your code goes here

    # Mean squared error equals ( Prediction of X - y value ) ** 2

    for i in range(len(predictions)):
        mean_sq = (predictions[i] - net_worths[i]) ** 2
        tuple_append = (ages[i][0], net_worths[i][0], mean_sq[0])
        temp_list.append(tuple_append)

    temp_list.sort(key=itemgetter(2))

    for i in range(81):
        cleaned_data.append(temp_list[i])

    return cleaned_data

