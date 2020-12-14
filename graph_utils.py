#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import os

from cell import utils, math_utils
import numpy as np
import pandas as pd
import networkx as nx
import random
from collections import Counter
from sinkhorn_knopp import sinkhorn_knopp as skp

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

def remove_nodes_from_edglist(edgelist, node_list):
    '''

    Parameters
    ----------
    edgelist
    node_list: a list of nodes to be removed

    Returns
    -------
    '''
    return edgelist[(~edgelist.source.isin(node_list)) & (~edgelist.target.isin(node_list))]


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

def return_weight_mat_from_edgelist(edgelist, directed=False):
    '''
    Gets the edgelist and return the weight matrix
    '''
    elongate_edge_list = []
    for idx, row in edgelist.iterrows():
        elongate_edge_list.append([row['source'],row['target'], row['weight']])
        if not directed:
            elongate_edge_list.append([row['target'],row['source'], row['weight']])
    elongate_edge_list = pd.DataFrame(elongate_edge_list, columns=['source', 'target', 'weight'])
    weight_mat = pd.pivot_table(elongate_edge_list, columns="target", index="source", values="weight")
    weight_mat[weight_mat.isnull()] = 0

    return weight_mat



def return_theta(n_nodes):
    '''
    Take number of nodes and create polar angles for plotting purposes
    Args:
        n_nodes: number of nodes in the graph

    Returns:

    '''
    return [ i * 2 * np.pi / n_nodes for i in range(n_nodes)]


def split_train_test_graph(adj, test_percent):
    '''
     parameters
    ----------
    adj: adjacency or the weight matrix as an np.array
    test_percent: total amount of edges to be saved for test

     return
    ----------
    train_adj: np.array
    test_adj: np.arary
    test_nodes: np.array
    '''
    train_adj = []
    test_adj = []
    test_nodes = []

    n_nodes = adj.shape[0]
    for i, row in enumerate(adj):
        select_random_index = np.random.choice(range(n_nodes), int(test_percent * 100))
        test_adj.append([row[j] for j in select_random_index])
        test_nodes.append(select_random_index)
        train_adj.append([0. if i in select_random_index else r for i, r in enumerate(row)])

    return np.array(train_adj), np.array(test_adj), np.array(test_nodes)


def get_frequency_tables(walks, node_list):
    """

    parameters
    ----------
    walks: list of list of walks
    node_list: list of node names

    return
    ----------
    frequency table for each node
    """

    freq = {k:Counter() for k in node_list}

    for walk in walks:
        n_i = walk[0]
        freq[n_i].update(walk[1:])
    return freq

def get_probability_from_freq_tables(frequeny_dict):
    """

    parameters
    ----------
    walks: list of list of walks
    node_list: list of node names

    return
    ----------
    frequency table for each node
    """
    prob = {}
    total = sum(frequeny_dict.values(), 0.0)
    # print(total)
    for key in frequeny_dict:
        prob[key] = frequeny_dict[key] / total
    return prob


def get_negative_probability_from_probability(probability, softmax_T=1):
    """

    parameters
    ----------
    probability:
    softmax_T:

    return
    ----------
    """
    negative_prob = {}
    for key in probability:
        negative_prob[key] = 1 - probability[key]

    # Apply softmax for forward
    for key in negative_prob:
        negative_prob[key] = np.exp(negative_prob[key] / softmax_T)

    total = sum(negative_prob.values(), 0.0)
    for key in negative_prob:
        negative_prob[key] /= total
    return negative_prob


def return_soccer_weight_matrix():
    '''
    It return a toy model of a soccer graph with 12 nodes, you can manuallu changes
    weights below if you wish
    '''

    edges = pd.DataFrame([['1', '2', 0.8],
                          ['1', '3', 0.1],
                          ['1', '4', 0.05],
                          ['1', '7', 0.05],
                          ['2', '6', 0.7],
                          ['2', '9', 0.3],
                          ['3', '9', 0.5],
                          ['3', '7', 0.5],
                          ['4', '8', 0.95],
                          ['4', '5', 0.05],
                          ['5', '1', 1],
                          ['6', '9', 0.2],
                          ['6', '10', 0.8],
                          ['7', '10', 0.15],
                          ['7', '11', 0.85],
                          ['8', '5', 0.55],
                          ['8', '4', 0.45],
                          ['9', '10', 1],
                          ['10', '11', 0.6],
                          ['10', '6', 0.4],
                          ['11', '8', 0.1],
                          ['11', '5', 0.9]], columns=['source', 'target', 'weight'])

    weight_mat = return_weight_mat_from_edgelist(edges, directed=True)
    weight_mat = weight_mat.loc[[str(i) for i in range(1, 12)]][[str(i) for i in range(1, 12)]]

    return weight_mat


def return_a_toy_graph_weight_mat(n_nodes, min_connection, max_connection):
    '''
    takes number of nodes and min and max number of connections per node and generate a
    toy graph weight matrix with random connections

    Args:
    -----
    n_nodes
    min_connection
    max_connection

    return
    ------
    weight mat

    '''
    total_inter = np.zeros((n_nodes, n_nodes))

    for irow, row in enumerate(total_inter):
        # generate index of random connections
        n_connections = random.sample(range(min_connection, max_connection), 1)
        con = random.sample([t for t in range(0, n_nodes) if t != irow], n_connections[0])
        # generate weights of random connections that adds up to 1
        wei = np.random.dirichlet(np.ones(n_connections), size=1)[0]
        for ic, c in enumerate(con):
            total_inter[irow][[c]] = wei[ic]

    total_inter = pd.DataFrame(total_inter)
    return total_inter





def Keep_only_k_largest_value_of_each_row(mat, k):
    '''
    Take a data frame, for each row it sorts the values of that row and only keeps
    the highest k member for each row and the rest will be set to zero

    Args:
    -----
    mat: A data frame of adj matrix for example
    k: how many members from each row should be kept, the rest will be set to zero
    '''
    df = mat.copy()
    df.index.name = 'index'
    melted = pd.melt(df.reset_index(), id_vars='index')
    melted.columns = ["row", 'col', 'value']
    melted["rank"] = melted.groupby("row")["value"].rank("dense", ascending=False)
    melted['keep'] = np.where(melted['rank'] <= k, melted['value'], 0.)
    new_df = melted.pivot(index='row', columns='col', values='keep')
    new_df = new_df.loc[[i for i in df.index.tolist()]][[i for i in df.columns.tolist()]]

    return new_df

def Keep_only_k_largest_value_of_each_row_and_each_column(mat, k):
    '''
    Take a data frame, for each row it sorts the values of that row and only keeps
    the highest k member for each row and the rest will be set to zero

    Args:
    -----
    mat: A data frame
    k: how many members from each row and column should be kept, the rest will be set to zero
    '''
    df = mat.copy()
    df.index.name = 'index'
    melted = pd.melt(df.reset_index(), id_vars='index')
    melted.columns = ["row", 'col', 'value']
    melted["rank_row"] = melted.groupby("row")["value"].rank("dense", ascending=False)
    melted["rank_col"] = melted.groupby("col")["value"].rank("dense", ascending=False)
    melted['keep'] = np.where((melted['rank_row'] <= k) | (melted['rank_col'] <= k), melted['value'], 0.)
    melted = melted.pivot(index='row', columns='col', values='keep')
    melted = melted.loc[[i for i in df.index.tolist()]][[i for i in df.columns.tolist()]]
    return melted

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


def apply_doubly_stochastic(adj):
    '''
    Args:
    -----
    adj: a dataframe that we want to make doubly stochastic

    return:
    -------
    ds: a doubly stochastic dataframe
    '''

    # First add a small value to members to get rid of zeros
    df = adj.copy()
    df = df + 0.00001

    sk = skp.SinkhornKnopp()
    # Apply SinkhornKnopp
    ds = sk.fit(df)
    ds = pd.DataFrame(ds)
    ds.index = df.index.astype(str)
    ds.columns = df.columns.astype(str)
    return ds


def get_column_index_of_sorted_rows(df):
    '''
    get a dataframe and sort each row and return the column index of each row in
    a new dataframe
    '''
    top_n = df.shape[0]
    sorted_rows_df = pd.DataFrame({n: df.T[col].nlargest(top_n).index.tolist() for n, col in enumerate(df.T)}).T
    sorted_rows_df.index = df.index
    return sorted_rows_df


def mask_allbut_k_percentile_of_each_row(df, percentile):
    '''
    get the df, sort each row and find the sorted row. For each row compute
    the sum and find the sum*percentile value as a traget. Then for each row
    keep the highest values that adds up to that target value. Then return a
    dictionary that has the keys as index of the df and the values are the column
    that should be kept for that row of df
    '''
    data = df.copy()
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)

    sorted_column_data = get_column_index_of_sorted_rows(data)
    sorted_column_data.index = sorted_column_data.index.astype(str)
    sorted_column_data.columns = sorted_column_data.columns.astype(str)

    data['sum'] = data.sum(axis=1)
    data['top_percentile'] = data['sum'] * percentile


    mask = sorted_column_data.stack().reset_index()
    mask = mask.rename(columns={"level_0": "index", "level_1": "keep", 0: "column"}, errors="raise")
    mask['keep'] = 0
    mask['column'] = mask['column'].astype(str)
    mask['index'] = mask['index'].astype(str)

    for idx, row in sorted_column_data.iterrows():
        cum = 0
        for value in row:
            cum += data.loc[idx][value]
            if cum <= data.loc[idx]['top_percentile']:
                mask.loc[(mask['index'] == str(idx)) & (mask['column']== str(value)), 'keep'] = 1

    mask = pd.pivot(mask, columns="column", index="index", values="keep")
    return mask

def mask_allbut_k_percentile_of_each_col(df, percentile):
    '''
    Look at the keep_k_percentile_of_each_row function for more details
    '''
    data = df.copy()
    data = data.T
    mask = mask_allbut_k_percentile_of_each_row(data, percentile)
    return mask.T

def keep_k_percentile_of_each_col_and_each_row(df, percentile):
    final_df = df.copy()
    final_df.index = final_df.index.astype(str)
    final_df.columns = final_df.columns.astype(str)
    mask_row = mask_allbut_k_percentile_of_each_row(final_df, percentile)
    mask_col = mask_allbut_k_percentile_of_each_col(final_df, percentile)
    mask_all = mask_row + mask_col
    mask_all = mask_all > 0
    final_df = final_df.where(mask_all, 0)
    final_df = final_df.rename_axis(None, axis=0)
    final_df = final_df.rename_axis(None, axis=1)

    return final_df

def keep_k_percentile_of_each_row(df, percentile):
    final_df = df.copy()
    mask_row = mask_allbut_k_percentile_of_each_row(final_df, percentile)
    mask_row = mask_row > 0
    final_df = final_df.where(mask_row, 0)
    final_df = final_df.rename_axis(None, axis=0)
    final_df = final_df.rename_axis(None, axis=1)

    return final_df
