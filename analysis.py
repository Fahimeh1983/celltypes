import pandas as pd

from cell import utils


def summarize_walk_embedding_results(gensim_dict, ndim, cl_df=None):
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
        node_ids = v.wv.index2word  # list of node IDs
        weighted_node_embeddings = v.wv.vectors  # the embedding vectors
        columns = ["Z" + str(i) for i in range(ndim)]  # new column names for the embedding vectors
        emb = pd.DataFrame(weighted_node_embeddings, index=node_ids, columns=columns)  # make a df out of embeddings
        emb = utils.Reset_Rename_index(emb, name="cluster_id")  # set the cluster_ids or node ids
        emb['cluster_id'] = emb['cluster_id'].apply(str)

        if cl_df is not None:
            cl_df['cluster_id'] = cl_df['cluster_id'].apply(str)
            emb = emb.merge(cl_df[["cluster_id", "cluster_color", "cluster_label"]], on="cluster_id")

        emb['channel_id'] = k
        data = data.append(emb)

    return data
