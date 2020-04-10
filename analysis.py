import pandas as pd
import numpy as np

from cell import utils
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def summarize_walk_embedding_results(gensim_dict, index, ndim, cl_df=None):
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