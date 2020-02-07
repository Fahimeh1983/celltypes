#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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