#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import os
import csv

import numpy as np
import pandas as pd

from enum import Enum
from IPython.display import Image


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
    if os.path.isdir('/home/fahimehb/Documents/'):
        network_root = '/home/fahimehb/Documents/'
    return network_root


def get_all_folder_names(path):
    output = [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
    return output

def get_all_file_names(path):
    output = os.listdir(path)
    return output

def get_npp_visp_interaction_mat_path(layer, version):
    roi = "VISp"
    root = get_dir_root()
    layer_filename = layer + ".csv"
    folder = "Interaction_mats_" + version
    path = concat_path(root, "NPP_GNN_project", "dat", folder, roi, layer_filename)
    # print("reading from:", path)
    return path

def get_npp_visp_layers(version):
    roi = "VISp"
    root = get_dir_root()
    filename = "layers_" + version + ".csv"
    path = concat_path(root, "NPP_GNN_project", "dat", filename)
    layers = pd.read_csv(path)['layers'].tolist()
    return layers

def get_walk_dir(roi, project_name, N, length, p, q, layer_class, layer, walk_type):
    root = get_dir_root()
    walk_dir_name = get_walk_dir_name(N, length, p, q)
    path = concat_path(root, project_name, "dat", "walks", roi, layer_class, walk_type, walk_dir_name, layer)
    return path

def get_walk_dir_name(N, length, p, q):

    if p % 1 == 0:
        p = str(int(p))
    else:
        p = f"{p:f}"

    if q % 1 == 0:
        q = str(int(q))
    else:
        q = f"{q:f}"

    return "_".join(("N", str(N), "l", str(length), "p", p, "q", q))

def get_model_dir(project_name, roi, N, length, p, q, layer_class, layer, walk_type):
    root = get_dir_root()
    walk_dir_name = get_walk_dir_name(N, length, p, q)
    path = concat_path(root, project_name, "models", roi, layer_class, walk_type, walk_dir_name, layer)
    return path

def get_model_name(size, iter, window, lr, batch_size, opt_add=None):
    filename = "_".join(("model", "size", str(size), "iter", str(iter), "window", str(window), "lr", str(lr),
                        "bs", str(batch_size)))
    if opt_add:
        filename = "_".join((filename, opt_add))
    filename = ".".join((filename, "csv"))
    return filename

def get_edgelist_dir(roi, project_name, layer):
    root = get_dir_root()
    path = concat_path(root, project_name, "dat", "edgelists", roi, layer)
    return path

def get_intraction_matrix_filename(roi, project_name, layer):
    root = get_dir_root()
    path = concat_path(root, project_name, "dat", "Interaction_mats", roi, layer)
    return ".".join((path, "csv"))

def get_loss_filename(size, iter, window, lr, batch_size, opt_add=None):
    filename = "_".join(("loss", "size", str(size), "iter", str(iter), "window", str(window), "lr", str(lr),
                         "bs", str(batch_size)))
    if opt_add:
        filename = "_".join((filename, opt_add))
    filename = ".".join((filename, "csv"))
    return filename


def get_E_R_embedding_filepath(filepath, prefix_filename, embedding_size, window, lamda):
    '''
    returns 2 filepaths for right and left embeddings

    Parameters
    ----------
    filepath: path
    prefix_filename: prefix
    embedding size
    window
    lamda

    Returns
    -------
    filepath for E and R embeddings
    '''

    filename =  "w" + str(window) + "_" + str(embedding_size) + "d.csv"
    E_path = filepath + prefix_filename + "lambda" + str(lamda) + "_E_" + filename
    R_path = filepath + prefix_filename + "lambda" + str(lamda) + "_R_" + filename
    print(E_path)
    print(R_path)
    return E_path, R_path

def read_npp_interaction_matrices(version, which_layers=None):
    '''
    Read the interaction matrices of npp and return the combined adj matrix
    if which_layer is none then it read all the layers and combine them

    Args:
    ----
    which_layers: a list of layer names that we want to read and combine
    '''

    npp_adj = np.zeros((93, 93))
    if which_layers is None:
        which_layers = get_npp_visp_layers(version)

    for layer in which_layers:
        path = get_npp_visp_interaction_mat_path(version=version, layer=layer)
        tmp_inter = pd.read_csv(path, index_col="Unnamed: 0")
        npp_adj = npp_adj + tmp_inter.values

    return npp_adj

def read_npp_interaction_df(version, which_layers, index=None, columns=None):
    '''
    Read the interaction matrices of npp and return the combined adj matrix
    if which_layer is none then it read all the layers and combine them

    Args:
    ----
    which_layers: a list of layer names that we want to read and combine
    '''

    df = read_npp_interaction_matrices(version, which_layers)
    df = pd.DataFrame(df, index=index, columns=columns)

    return df


#################################################
#
#     Read / writes / show
#
#################################################

def read_visp_npp_cldf(version):
    root = get_dir_root()
    if version == "old":
        path = concat_path(root, "NPP_GNN_project", "dat", "cl_df_VISp_annotation.csv")
    if version == "new":
        path = concat_path(root, "NPP_GNN_project", "dat", "cl_df_VISp_annotation_updated.csv")
    cl_df = pd.read_csv(path, index_col="cluster_id")
    print("Reading cldf from:", path)
    cl_df.index = cl_df.index.astype(str)
    return cl_df

def _raise_error(msg):
    raise ValueError("({}) {}".format(type().__name__, msg))


def read_list_of_lists_from_csv(path):

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

    with open(path , 'r') as f:
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=','))

    return data

def write_list_of_lists_to_csv(path, file):

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

    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(file)

    print("Done writing!")

def write_list_to_csv(path, file):
    with open(path, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(file)
    print("Done writing!")

def read_list_from_csv(path):
    with open(path, 'r') as myfile:
        reader = csv.reader(myfile)
        data = list(reader)
    return data[0]

def print_whole_df(df):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    return df



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

def get_norm(coords):
    '''
    get coordiantes in an np.array and compute the L2 norm of each row
    Parameters
    ----------
    coords

    Returns
    -------

    '''
    norm = []
    for coord in coords:
        norm.append(np.linalg.norm(coord))
    return np.array(norm)


def normalized_each_row_by_rowsum(df):
    '''
    get a df and divide each row by the rowsum

    Parameters
    ----------
    df

    Returns
    -------
    '''
    row_sums = df.sum(axis=1)
    new_matrix = df / row_sums[:, np.newaxis]
    return new_matrix

def shift_E_R_to_positive(paths):
    '''
    move E and R embeddings to positive values

    Parameters
    ----------
    filepath: list of E and R paths

    Returns
    -------
    shifted coordinates
    '''
    mins = []
    for path in paths:
        file = pd.read_csv(path, index_col="Unnamed: 0")
        # get the min values for all columns
        mins.append(np.array(np.min(file, axis=0)))

    min_mins = np.min(np.array(mins), axis=0)

    shifted_files = []
    for path in paths:
        file = pd.read_csv(path, index_col="Unnamed: 0")
        shifted_files.append(file - min_mins)
    return shifted_files