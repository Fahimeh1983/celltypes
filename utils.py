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

def Reset_Rename_index(df, name, index_col_name= None):

    """
    set the index to a new column and rename the column

    Parameters
    ----------
    df: Data frame
    name: the new colname for the index
    index_col_name: some times index column has a name which is
    different than "index", in that case provide the name of the index col as
    of now

    Returns
    -------
    df: data frame with a new column which was previously the index
    """
    if index_col_name is not None:
        df = df.reset_index().rename(columns={index_col_name: name})
    else:
        df = df.reset_index().rename(columns={"index": name})

    return df

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

