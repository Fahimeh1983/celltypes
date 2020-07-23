import pandas as pd
import numpy as np

from cell import utils
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF
from sklearn.metrics import ndcg_score, dcg_score
from scipy import linalg
import scipy as sp
from statistics import mean

def summarize_walk_embedding_results(gensim_dict, index, ndim, cl_df=None, padding_label=None):
    """
    Takes a dictionary of gensim word2vec output and make a data frame for more analysis. This can be used
    for only one or for multiple graphs. so if it is only one graph then the dict has only one key

    Parameters
    ----------
    gensim_dict: keys are the name or the label of the graph and the values are the word2vec output
    ndim: embedding size
    cl_df: it is a reference df which has the cluster_id and cluster_color, if provided, then those data
    will be added to the output data frame

    Returns
    -------
    a data frame which has the embedding and some more info if cl_df is provided
    """

    data = pd.DataFrame()

    for k, v in gensim_dict.items():

        emb = pd.DataFrame(v, index=index)
        emb.columns = ["Z" + str(i) for i in range(ndim)]
        emb.index.name = "cluster_id"

        if cl_df is not None:
            if padding_label is None:
                raise ValueError("When cl_df is given, padding_idx must be provided")
            else:
                emb = emb.drop(padding_label)

            cl_df.index = cl_df.index.astype(str)
            emb = emb.merge(cl_df, on="cluster_id")

        emb['channel_id'] = k
        data = data.append(emb)

    return data



def get_closest_nodes(emb, index, node, topn):
    '''

    Parameters
    ----------
    emb:  emb or coordinate for each node
    index: index of nodes that emb belongs to
    node: the node that you want its closest neighbors
    topn: number of neighbors to print
    Returns
    -------
    a list of closest nodes to the given node
    '''

    dist = squareform(pdist(emb))
    dist = pd.DataFrame(dist, index=index, columns=index)
    dist.values[np.arange(dist.shape[0]),np.arange(dist.shape[0])] = dist.max(axis=1)
    idx = np.argsort(dist.loc[node].tolist())
    sorted_indices = [index[i] for i in idx]
    return sorted_indices[0:topn]

def get_closest_node_label(emb, index, node, topn, cl_df):
    '''

    Parameters
    ----------
    emb
    index
    node
    cl_df
    topn

    Returns
    -------

    '''
    nn = get_closest_nodes(emb, index, node, topn)
    print("closest nodes to :", cl_df.loc[[node]]['cluster_label'][0])
    print("------------------------------------------")
    print("")
    print(cl_df.loc[[i for i in nn]][['cluster_label']])
    return cl_df.loc[[i for i in nn]]['cluster_label'].tolist()


def run_procrustes_analysis(df1, df2, cl_df=None):
    '''
    reindex df2 based on df1 indices and run procrustes analysis
    The indexes must be the cluster_ids

    Parameters
    ----------
    df1: df1
    df2: df2
    cl_df(optional): if provided, the fuction will return a merge datafram including all cl_df info
    Returns
    -------

    '''

    if df1.shape != df2.shape:
        raise ValueError("Two data frames should have same dimensions")

    df2 = df2.reindex(df1.index)
    Z_cols = [col for col in df1.columns if 'Z' in col]

    mtx1, mtx2, disparity = procrustes(df1[Z_cols], df2[Z_cols])
    mtx1 = pd.DataFrame(mtx1, index=df1.index, columns=Z_cols)
    mtx2 = pd.DataFrame(mtx2, index=df2.index, columns=Z_cols)
    if cl_df is not None:
       mtx1 = mtx1.merge(cl_df, on="cluster_id")
       mtx2 = mtx2.merge(cl_df, on="cluster_id")

    if 'cluster_color' in df1.columns:
        mtx1['cluster_color'] = df1['cluster_color']
        mtx2['cluster_color'] = df1['cluster_color']

    return mtx1, mtx2, disparity

def Katz_proximity(adj, beta):
    '''
    Take the adj matrix and beta and compute Katz
    Parameters
    ----------
    adj
    beta: decay factor must be smaller than the spectral radius

    Returns
    -------

    '''
    M_g = np.eye(adj.shape[0], adj.shape[1]) - beta * adj
    M_l = beta * adj
    inverse_M_g = np.linalg.inv(M_g)
    return np.dot(inverse_M_g, M_l)

def nmf_factorization(matrix, n_components):
    '''
    Takes a matrix and factorize it using NMF
    Parameters
    ----------
    matrix
    n_components

    Returns
    -------

    '''
    model = NMF(n_components=n_components, init='random', random_state=0)
    U_s= model.fit_transform(matrix)
    U_t = np.transpose(model.components_)
    return U_s, U_t

def svd_factorization(matrix, full_matrices):
    U, s, Vh = linalg.svd(matrix, full_matrices=full_matrices)
    S = np.diag(s)
    return U, np.dot(S, Vh)

def return_right_left_similarity(U_s, U_t):
    '''
    take right and left eigen vectors and compute the dot products or similarity
    Args:
        U_s: source vector embeddings
        U_t: target vector embeddings

    Returns:
    left and right similarity matrices
    '''
    s_to_t = np.dot(U_s, U_t.T)
    t_to_s = np.dot(U_t, U_s.T)

    return s_to_t, t_to_s

def NRMSE(mat, U_s, U_t):
    '''
     parameters
    ----------
    mat: The matrix that was factorized using nmf form example
    U_s: The right eigen vector obtained by factorization of mat
    U_t: The left eigen vector obtained by factorization of mat

     return
    ----------
    NRMSE: returns the normalized RMSE as in HOPE paper
    '''

    Fro_norm = sp.linalg.norm(mat - np.dot(U_s, U_t.T), 'fro')
    num = np.power(Fro_norm, 2)

    Fro_norm = sp.linalg.norm(mat, 'fro')
    denum = np.power(Fro_norm, 2)

    return np.sqrt(num / denum)


def nandcg_score_at_k(similarity, adj, k=None):
    '''
    node_averaged_ndcg

    Parameters
    ----------
    similarity: np.array of similarity scores, each row is the similarity of that node with respect to all other nodes
    adj: adjacency np.array, the weights will be used as the relevance for each edge
    k: int, if provided, the ndcg at rank k will be used. If none all the ranks will be used

    Returns
    -------
    nandcg_score_at_k: np.array, averaged ndcg over all nodes of the graph
    '''

    if adj.shape[0] != adj.shape[1]:
        utils.raise_error(
            "adj should be an square np.array, instead is ({}) by ({})".format(adj.shape[0], adj.shape[1])
        )

    ndcg = []
    n_nodes = adj.shape[0]
    for n in range(n_nodes):
        sorted_similarity_index_of_n = np.argsort(-similarity[n])
        predicted_scores = np.array([similarity[n][sorted_similarity_index_of_n]])
        true_relevance = np.array([adj[n][sorted_similarity_index_of_n]])
        ndcg.append(ndcg_score(true_relevance, predicted_scores, k=k))

    return sum(ndcg) / n_nodes

def pairwise_ucleadian_dist_between_rows(a):
    '''
    find ucleadian distance between each row of an np.array

    Parameters
    ----------
    a: np.array

    Returns
    -------
    ucleadian distance between each row of an np.array
    '''
    b = a.reshape(a.shape[0], 1, a.shape[1])
    return np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))


def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

