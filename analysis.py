import pandas as pd

from cell import utils


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
