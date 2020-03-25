#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import os
import csv

import numpy as np
import pandas as pd

from enum import Enum

################################################
#
#     Enumerated constant values
#
#################################################

class Name(Enum):
    def __str__(self):
        return str(self.value)

class ROI(Name):
    VISp = 'visp'
    ALM = 'alm'

#################################################
#
#     Paths / filenames
#
#################################################

def concat_path(*args):
    """ Join paths together by parts, worry-free """
    clean = [str(s).strip('/') for s in args]
    clean = ["/"] + clean
    return '/'.join(clean)

def get_dir_root():
    """ For dealing with different netweork locations on Mac/Linux """
    if os.path.isdir('/Users/fahimehb/Documents/'):
        network_root = '/Users/fahimehb/Documents/'
    if os.path.isdir('/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/'):
        network_root = '/allen/programs/celltypes/workgroups/rnaseqanalysis/Fahimehb/'
    if os.path.isdir('/home/pogo/work_dir/'):
        network_root = '/home/pogo/work_dir/'
    return network_root


def get_all_folder_names(path):
    output = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
    return output

def get_all_file_names(path):
    output = os.listdir(path)
    return output

def get_npp_visp_interaction_mat_path(layer):
    roi = "VISp"
    root = get_dir_root()
    layer_filename = layer + ".csv"
    path = concat_path(root, "NPP_GNN_project", "dat", "Interaction_mats", roi, layer_filename)
    print(path)
    return path

def get_npp_visp_layers():
    roi = "VISp"
    root = get_dir_root()
    path = concat_path(root, "NPP_GNN_project", "dat", "graphs", roi)
    layers = get_all_folder_names(path)
    print("listing from:", path)
    return layers

def get_walk_dir(roi, length, p, q, layer_class, walk_type):
    root = get_dir_root()
    walk_dir_name = get_walk_dir_name(length, p, q)
    path = concat_path(root, "NPP_GNN_project", "dat", "walks", roi, layer_class, walk_type, walk_dir_name)
    return path

def get_walk_dir_name(length, p, q):
    return "_".join(("l", str(length), "p", str(p), "q", str(q)))

def get_model_dir(roi, length, p, q, layer_class, walk_type):
    root = get_dir_root()
    walk_dir_name = get_walk_dir_name(length, p, q)
    path = concat_path(root, "NPP_GNN_project", "models", roi, layer_class, walk_type, walk_dir_name)
    return path

def get_loss_filename(size, iter, window, min_count, sg):
    filename = "_".join(("size", str(size), "iter", str(iter), "window", str(window), "mincount", str(min_count), "sg",
                    str(sg)))
    filename = ".".join((filename, "csv"))
    return filename

#################################################
#
#     Read / writes
#
#################################################

def read_visp_npp_cldf():
    root = get_dir_root()
    path = concat_path(root, "NPP_GNN_project", "dat", "cl_df_VISp_annotation.csv")
    cl_df = pd.read_csv(path, index_col="cluster_id")
    print("Reading cldf from:", path)
    cl_df.index = cl_df.index.astype(str)
    return cl_df

def _raise_error(msg):
    raise ValueError("({}) {}".format(type().__name__, msg))


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

def write_list_of_lists_to_csv(path, filename, file):

    """
    Write a list of lists to a csv file

    Parameters
    ----------
    path : a path to file
    filename : a string which is basically the filename.

    return:
    ----------
    list of list
    """

    with open(path + "/" + filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(file)

    return print("Done writing!")

#################################################
#
#     file manipulations
#
#################################################

def reset_rename_index(df, name, index_col_name= None):

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
