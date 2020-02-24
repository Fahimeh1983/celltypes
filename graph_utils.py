#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

from cell import utils
import numpy as np
import pandas as pd
import networkx as nx

def Graph_from_Weight_Matrix(weight_matrix, threshold, directed):
    '''
    Takes weight matrix and threshold(optional) and creates a graph(diredted) or non-directed

    parameters
    ----------
    weight_matrix : a squared matrix or data frame which has values as weights
    threshold: it must be None if you do not want to threshold the matrix
    directed : True or False

    returns
    ----------
    Graph object
    '''

    symmetric = utils.Check_Symmetric(weight_matrix)

    if threshold is not None:
        weight_matrix = weight_matrix[weight_matrix > threshold]

    if symmetric:
        print("The weight matrix is symmetric!")
        print("Upper triangle and diag are used as weights")
        upper_tri = np.triu(np.ones(weight_matrix.shape), 0).astype(np.bool)
        node_edge_weight = weight_matrix.where(upper_tri)
        node_edge_weight = node_edge_weight.stack().reset_index()
        node_edge_weight.columns = ['node1', 'node2', 'weight']

    else:
        print("The weight matrix is non-symmetric!")
        node_edge_weight = weight_matrix.stack().reset_index()
        node_edge_weight.columns = ['node1', 'node2', 'weight']

    if symmetric:
        if directed:
            raise ValueError('When w is symmetric, you can not make a directed G!')
        else:
            G = nx.Graph()

    if not symmetric:
        if not directed:
            raise ValueError('W is not symmetric, you can not make a non-directed G!')
        else:
            G = nx.DiGraph()

    G.add_weighted_edges_from([tuple(x) for x in node_edge_weight.values])
    return G


def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals