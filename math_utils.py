#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import random
from random import random
import numpy as np
import pandas as pd


def Linear_regression_model(X, y, test_size, random_state):

    """
    Perform linear regression and returns the prediction and error

    parameters
    ----------
    X: a data frame with the x values
    y: a data frame with the labels of x values, the rows should be the same
    order as x
    testsize: number or percentage of test data
    random_state : random state for partitioning the data to train and test

    return
    ----------
    y_test: the real y which was not seen during regression
    y_predict: the predicted y value
    RMSE: Root mean squared error on y prediction
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_predict = reg.predict(X_test)

    RMSE = np.sqrt(metrics.mean_squared_error(np.array(y_test), y_predict))

    return y_test, y_predict, RMSE

def intersection(lst1, lst2):

    """
    Find intersection of two lists

    Parameters
    ----------
    lst1 : first list
    lst2 : second list

    Returns
    -------
    lst3 : a list of intersection values

    """

    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def Check_Symmetric(a, rtol=1e-05, atol=1e-08):

    """
    Checks if a matrix is symmetric or not

    Parameters
    ----------
    a: a matrix or data frame or numpy array

    returns:
    ----------
    TRUE or FALSE
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)




def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects',
        the likelihood of the objects is weighted according
        to the sequence of 'weights', i.e. percentages."""

    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]