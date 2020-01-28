#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import numpy as np
import pandas as pd
import csv



def Read_List_of_Lists_from_CSV(path, filename):

    """
     Read and return a list of lists from a csv file

    Parameters
    ----------
    path : a path to file
    filename : a string which is basically the filename.

    Returns
    -------
    data : a list of lists
    """

    with open(path + filename, 'r') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=','))

    return data

def Write_List_of_Lists_from_CSV(path, filename, file):

    """
     Write a list of lists to a csv file

    Parameters
    ----------
    path : a path to file
    filename : a string which is basically the filename.
    object: list of list object to be written into a csv file

    """

    with open(path + filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(file)

    return print("Done writing!")

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

def Divide_each_Row_by_colsum(df):
    """
         Divide each row by column sum

        Parameters
        ----------
        df: Data frame

        Returns
        -------
        df_normal: return the column-sum normalized df

        """
    df_normal = df.div(df.sum(axis=1), axis=0)
    return df_normal