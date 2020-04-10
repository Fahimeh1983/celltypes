#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import os

from cell import utils, math_utils
import numpy as np
import pandas as pd
import networkx as nx

from stellargraph import StellarDiGraph, StellarGraph


def check_wts(graph, weighted):
    '''
    Check the weights of a stellar graph object

    parameters
    ----------
    graph: a stellar graph object
    weighted : True or False

    returns
    ----------

    '''

    if weighted:
        # Check that all edge weights are greater than or equal to 0.
        # Also, if the given graph is a MultiGraph, then check that there are no two edges between
        # the same two nodes with different weights.
        for node in graph.nodes():
            # TODO Encapsulate edge weights
            for out_neighbor in graph.out_nodes(node):

                wts = set()
                for weight in graph._edge_weights(node, out_neighbor):
                    if weight is None or np.isnan(weight) or weight == np.inf:
                        utils.raise_error(
                            "Missing or invalid edge weight ({}) between ({}) and ({}).".format(
                                weight, node, out_neighbor
                            )
                        )
                    if not isinstance(weight, (int, float)):
                        utils.raise_error(
                            "Edge weight between nodes ({}) and ({}) is not numeric ({}).".format(
                                node, out_neighbor, weight
                            )
                        )
                    if weight < 0:  # check if edge has a negative weight
                        utils.raise_error(
                            "An edge weight between nodes ({}) and ({}) is negative ({}).".format(
                                node, out_neighbor, weight
                            )
                        )

                    wts.add(weight)
                if len(wts) > 1:
                    # multigraph with different weights on edges between same pair of nodes
                    utils.raise_error(
                        "({}) and ({}) have multiple edges with weights ({}). Ambiguous to choose an edge for the random walk.".format(
                            node, out_neighbor, list(wts)
                        )
                    )
    else:
        utils.raise_error("This check is done only for weighted graphs")



def Make_stellar_graph(path_to_nodes, path_to_edges, directed):
    '''
    Create a Stellar graph object from nodes and edges

    parameters
    ----------
    path_to_nodes: is the path to nodes dir. There should be a file names nodes.csv in that dir.
                    Which has a list of nodes, I think they must be strings
    path_to_edges : is the path to edges dir. There should be a file named edges.csv in that dir.
                    The edge file will have 3 columns with the names of "source", "target" and "weight"
    directed: True or False, if yes then it means that the edges are made from
              "source" to "target" and not the other way around

    returns
    ----------
    Graph object
    '''
    nodes_file = os.path.join(path_to_nodes, "nodes.csv")
    nodes = pd.read_csv(nodes_file, index_col="Unnamed: 0")

    edges_file = os.path.join(path_to_edges, "edges.csv")
    edges = pd.read_csv(edges_file, index_col="Unnamed: 0")

    if directed:
        print("Directed graph is made")
        Gs = StellarDiGraph(nodes, edges)
    else:
        print("UN-Directed graph is made")
        Gs = StellarGraph(nodes, edges)

    return edges


def build_edge_list(weight_matrix, threshold, directed):
    '''
    Takes weight matrix and threshold(optional) and creates a graph(diredted) or non-directed

    parameters
    ----------
    weight_matrix : a squared matrix or data frame which has values as weights
    threshold: it must be None if you do not want to threshold the matrix
    directed: If it is undirected, then it requires only the upper half of the
              matrix for the edges

    returns
    ----------
    a data frame of edge lists which has the source, target and weight
    '''

    symmetric = math_utils.Check_Symmetric(weight_matrix)
    weight_matrix.index = weight_matrix.index.astype(str)
    weight_matrix.columns = weight_matrix.columns.astype(str)

    if threshold is not None:
        weight_matrix = weight_matrix[weight_matrix > threshold]

    if not directed:
        if not symmetric:
            print("When undirected, the matrix must be symmetirc")
            exit()
        else:
            print("Upper triangle and diag are used as weights")
            upper_tri = np.triu(np.ones(weight_matrix.shape), 0).astype(np.bool)
            node_edge_weight = weight_matrix.where(upper_tri)
            node_edge_weight = node_edge_weight.stack().reset_index()
            node_edge_weight.columns = ['source', 'target', 'weight']


    else:
        print("Building a directed graph edge list")
        node_edge_weight = weight_matrix.stack().reset_index()
        node_edge_weight.columns = ['source', 'target', 'weight']


    return node_edge_weight


def get_node_from_edgelist(source_target_weight):
    """
    Takes the source target list and return all the nodes exists
    Returns
    -------

    """
    source_target_weight[['source', 'target']] = source_target_weight[['source', 'target']].astype(str)

    source_list = source_target_weight['source'].tolist()
    target_list = source_target_weight['target'].tolist()

    for i in target_list:
        source_list.append(i)

    all_nodes = list(set(source_list))
    return all_nodes


def fix_self_connection(source_target_weight, weighted):
    '''
       Takes the source, target, weight data frame and create a graph

       parameters
       ----------
       source_target_weight : a dataframe which has the source node, target node and the weight
       directed : True or False

       returns
       ----------
       Graph object
       '''

    nodes = get_node_from_edgelist(source_target_weight)
    source_target_weight[['source', 'target']] = source_target_weight[['source', 'target']].astype(str)

    for node in nodes:
        if source_target_weight[(source_target_weight['source'] == node)
                                & (source_target_weight['target'] == node)].empty:
            if weighted:
                small_w = np.random.random(1) * 10 ** -5
            else:
                small_w = 1

            source_target_weight = source_target_weight.append({"source": node,
                                                                "target": node,
                                                                "weight": small_w[0]}, ignore_index=True)

    source_target_weight[['source', 'target']] = source_target_weight[['source', 'target']].astype(str)

    return source_target_weight




def build_nx_graph(source_target_weight, directed):
    '''
    Takes the source, target, weight data frame and create a graph

    parameters
    ----------
    source_target_weight : a dataframe which has the source node, target node and the weight
    directed : True or False

    returns
    ----------
    Graph object
    '''


    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_weighted_edges_from([tuple(x) for x in source_target_weight.values])
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

