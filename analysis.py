import pandas as pd
import numpy as np

from cell import utils
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF
from sklearn.metrics import ndcg_score
from scipy import linalg
import scipy as sp
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


def summarize_embedding_results(E, R, resolution=None, cldf=None):
    """
    Args:
    ------
    E: df, emitter representation
    R: df, receiver representation
    resoltuion: "cluster_label", "subclass_label" or class_label
    cldf: the ref metadata

    Returns
    -------
    emit: emitter data frame with metadata attached
    rece: receiver data frame with metadata attached

    """
    emit = E.copy()
    rece = R.copy()

    emit.index = emit.index.astype(str)
    rece.index = rece.index.astype(str)

    emit['node_act'] = "E"
    rece['node_act'] = "R"
    # cols = emit.columns.tolist()

    if cldf is not None:
        ref = cldf.copy()
        ref = ref.reset_index()
        if resolution is None:
            raise ValueError("When cldf is given a resolution os required")

        if resolution == "cluster_label":
            resolution_id = "cluster_id"
            resolution_color = "cluster_color"
            emit.index.name = "cluster_id"
            rece.index.name = "cluster_id"

        if resolution == "subclass_label":
            resolution_id = "subclass_id"
            resolution_color = "subclass_color"
            emit.index.name = "subclass_id"
            rece.index.name = "subclass_id"

        if resolution == "class_label":
            resolution_id = "class_id"
            resolution_color = "class_color"
            emit.index.name = "class_id"
            rece.index.name = "class_id"

        ref[resolution_id] = ref[resolution_id].astype(str)
        # cols = cols + [resolution, resolution_color]

        # merge with ref metadata
        emit = emit.merge(ref, on=resolution_id)
        emit = emit.set_index(resolution_id, drop=True)

        rece = rece.merge(ref, on=resolution_id)
        rece = rece.set_index(resolution_id, drop=True)

    return emit, rece


def get_closest_nodes(emb, coordinate_cols, index, node, topn):
    '''

    Parameters
    ----------
    emb:  emb or coordinate for each node
    coordinate_cols: the columns that emb coordiantes are stored
    index: index of nodes that emb belongs to
    node: the node that you want its closest neighbors
    topn: number of neighbors to print
    Returns
    -------
    a list of closest nodes to the given node
    '''

    my_emb = emb.copy()
    emb.index = index
    my_emb = my_emb[coordinate_cols]
    my_emb.index = index
    dist = squareform(pdist(my_emb))
    dist = pd.DataFrame(dist, index=index, columns=index)
    closest_node_ids = dist.loc[node][np.argsort(dist.loc[node])][0:topn]
    emb.loc[closest_node_ids.index.tolist(), "dist"] = closest_node_ids

    return emb.loc[closest_node_ids.index.tolist()]


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
    similarity: np.array of similarity scores, each row is the similarity of that node with respect to all other nodes.
    The index and column must be the same as adj index and column
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
        predicted_scores = np.array([similarity[n]])
        true_relevance = np.array([adj[n]])
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


def get_distance_between_eachrow_of_one_df_with_all_rows_of_other_df(df1, df2):
    '''
    df1: first data frame
    df2: second data frame

    return:
    --------
    df1_to_df2: distance between each row of the first dataframe with all the all the rows of the second
    '''
    df1_to_df2 = pd.DataFrame(index=df1.index.to_list(), columns=df1.index.to_list())

    for idr1, row1 in df1.iterrows():
        for idr2, row2 in df2.iterrows():
            df1_to_df2.loc[idr1][idr2] = np.linalg.norm(row1 - row2)

    df1_to_df2.index = df1_to_df2.index.astype(str)
    df1_to_df2.columns = df1_to_df2.columns.astype(str)

    return df1_to_df2


def get_distance_node_importance(node_id, dist_df):
    '''
   get a distance matrix and for the given node_id, sort that row and find the importance of each node
   the closer the node to a node_id, the higher the importance

    Parameters
    ----------
    node_id: str
    dist_df : a data frame of distance values

    Returns
    -------
    dictinary with the all the nodes and their importance to the given node_id
    '''
    # get colnames of the negative sorted row for that given node_id (the closest node is the last in this list)
    cols = dist_df.columns[np.argsort(-dist_df.loc[node_id])]

    node_importance = {}
    for idx, node in enumerate(cols):
        node_importance[node] = idx + 1
    return node_importance


def get_distance_ndcg_score(node_id, E_to_R, adj, k):
    '''
    Get weight or adj matrix, note that in weight matrix all the edges are outgoing edges, so we compare that to E_to_R
    distance matrix and not R_to_E

    Parameters
    ----------
    node_id: str
    E_to_R : emitter to receiver distance matrix
    adj: outgoing adj matrix
    k: ndcg rank

    Returns
    -------
    returns ndcg score
    '''

    true_y = np.array([adj.loc[node_id].tolist()])
    node_imp_dict = get_distance_node_importance(node_id, E_to_R)
    predicted_y = np.array([[node_imp_dict[i] for i in adj.columns]])
    return true_y, predicted_y, node_imp_dict, ndcg_score(true_y, predicted_y, k=k)


def precision_recall_at_k(predictions, k=10):
    """Return precision and recall at k metrics for each user

    Args:
        predictions: a list of dictionaries, each dict has 4 keys, uid is the source node
        tid is the target node, true_rel is the adj matrix value or the true relevance of
        target node and source node, est is the estimated value of the true relevance using
        our machine learning model, which is basically similarity obtained by distance between
        the embedding of source and target. We obtain the coordinate, then distance and then
        for each node we subtarct the max value of each row from each value of the row. This
        way the node which has the lowest distance will have the highest value which we call
        it similarity. If threshold is zero, the absolute values are not important and we will
        only use the order to compute the presicion and recall

        k: is the rank k

    Returns:
       sqrt of the min of the svd in all direction of embedding space

    """

    #     threshold: float value for thresholding true relevance and estimated values. In our
    #     case, since estimated values are the similarity between the nodes obtained from the
    #     embedding and true relevance is the edge weight, we must put the threshold to zero
    #     which means we are taking all the non zero weight edges as relevant

    print("This measure makes sense for when the small weights are set to zero and everything more"
         "than 0.00001 is a relevant weight and almost equal to all other weights")

    threshold = 0.00001
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for prediction in predictions:
        user_est_true[prediction['uid']].append((prediction['est'], prediction['true_rel'], prediction['tid']))

    precisions = dict()
    recalls = dict()
    sorted_user_est_true = defaultdict(list)

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        sorted_user_est_true[uid] = user_ratings


        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r, _) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r, _) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        # print(n_rec_k, n_rel)

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls, sorted_user_est_true


def get_similarity_from_distance_matrix(e_to_r):
    '''
    Get the distance matrix which each row has the distance of that node from all the other
    nodes in the graph and compute the similarity. We define the similarity for each row as
    each value subtracted from the max value of that row and then we take the absolute value.
    This way the larget simialrity is the smallest distance


    Parameters
    ----------
    node_id: str
    E_to_R : emitter to receiver distance matrix

    Returns
    -------
    similarity
    '''

    df = e_to_r.copy()
    df['row_max'] = np.max(df, axis=1)
    similarity = df.sub(df['row_max'], axis="index").abs()
    similarity = similarity.drop("row_max", axis=1)
    return  similarity


def get_predictions_correct_format_for_precision_recall_function(estimated_relevance, true_relevance):
    '''
    gets two data frames of estimated relevance and the true relevance and returns
    a list of all predictions in correct format to feed to precisions recall function


    Parameters
    ----------
    estimated_relevance: a data frame which is estimated relevance or similairty between each two nodes.
    in our case we are not going to use the exact values, instead we just sort them based on the values
    as in precision recall calculations we have a threshold of zero
    true_relevance : a data frame which is true relevance between each two nodes. In our case we are using
    the adj values or edge weights but again we not using the absolute values but instead as the threshold
    in precision recall is zero, we are going to just care if it is zero or non-zero

    Returns
    -------

    '''
    nodes = estimated_relevance.index.tolist()
    predictions = []
    for s in nodes:
        for t in nodes:
            predictions.append({"uid": s,
                                "tid": t,
                                "true_rel": true_relevance.loc[s][t],
                                "est": estimated_relevance.loc[s][t]})

    return predictions


def Compute_node_average_ndcg(adj, e_to_r, k):
    '''
    Takes the true relevance matrix which can be adj matrix and the distance matrix
    between emitter and receiver represenation and compute the ndcg per node and then
    return the average

    Args:
    -----
    adj: True relevane or the adj matrix
    e_to_r: each row is the distance between each E node(index) and all the other R nodes(columns)
    k: Rank for computing ncdg
    '''

    nandcg = {}
    for n in e_to_r.index.tolist():
        _, _, _, nandcg[n] = get_distance_ndcg_score(n, e_to_r, adj, k=k)

    return np.mean([v for v in nandcg.values()])


def get_closest_nodes_info(E_to_R_dist, adjacency, topn, node_id=None, node_label=None, cldf=None, resolution=None, node_action="E"):
    '''
    Takes the E_to_R_dist matrix and the adj and prints the closest Receivers to each emitter and vice versa
    '''

    if node_label is not None:
        if node_id is not None:
            utils.raise_error("node id and node label can not both Not none")
        else:
            node_id = cldf[cldf['cluster_label'] == node_label].index[0]

    dist = E_to_R_dist.copy()
    adj = adjacency.copy()

    if node_action == "R":
        dist = dist.T
        adj = adj.T

    ref = cldf.copy()
    ref = ref.reset_index()
    ref['cluster_id'] = ref['cluster_id'].apply(str)
    ref['subclass_id'] = ref['subclass_id'].apply(str)
    ref['class_id'] = ref['class_id'].apply(str)

    df = pd.DataFrame()
    true_y, predicted_y, node_imp_dict, _ = get_distance_ndcg_score(node_id, dist, adj, topn)
    X = true_y[0]  # taken from the adj
    Y = predicted_y[0]  # computed distances and then scored, the smallet distance has the highest score

    # first sort the predicted_y from largest to smallest and then order the true_y based on the predicted_y
    Z = [x for _, x in sorted(zip(Y, X),reverse=True)]
    sorted_neighbors_scores = sorted(Y, reverse=True)[0:topn]
    print()

    df['predicted_neighbors_weights'] = Z[0:topn]

    if resolution is not None:
        if resolution == "cluster_label":
            resolution_id = "cluster_id"
        if resolution == "subclass_label":
            resolution_id = "subclass_id"
        if resolution == "class_label":
            resolution_id = "class_id"

    print("Closet neighbors of", resolution, ref[ref[resolution_id] == node_id][resolution].tolist()[0])
    print("______________________________________")


    index_list = []
    for s in sorted_neighbors_scores:
        index_list.append([k for (k, v) in node_imp_dict.items() if v == s][0])
    df['predicted_closest_neighbors_index'] = index_list

    if ref is not None:
        if resolution is None:
            utils.raise_error(
                "Resolution must be provided, cluster_label, subclass_label or class_label"
            )
        my_list = []
        for idx in df['predicted_closest_neighbors_index']:
            label = ref[ref[resolution_id] == idx][resolution].tolist()
            if resolution_id == "cluster_label":
                if len(label) > 1:
                    utils.raise_error(
                        "more than one cluster_label are presenet"
                    )
            my_list.append(label[0])
        df['predicted_closest_neighbors_label'] = my_list

    df['Actual_neighbor_weights'] = sorted(X, reverse=True)[0:topn]

    index_list = []
    for value in df['Actual_neighbor_weights']:
        index_list.append(adj.loc[node_id][adj.loc[node_id] == value].index.tolist()[0])
    df['Actual_neighbor_index'] = index_list


    if ref is not None:
        if resolution is None:
            utils.raise_error(
                "Resolution must be provided, cluster_label, subclass_label or class_label"
            )
        my_list = []
        for idx in df['Actual_neighbor_index']:
            label = ref[ref[resolution_id] == idx][resolution].tolist()
            if resolution_id == "cluster_label":
                if len(label) > 1:
                    utils.raise_error(
                        "more than one cluster_label are presenet"
                    )
            my_list.append(label[0])
        df['Actual_neighbor_label'] = my_list

    match = [1 if i in df['predicted_closest_neighbors_index'].tolist() else 0 for i in
             df['Actual_neighbor_index'].tolist()]
    df['match'] = match

    return df


def Compute_umap(emb, emb_size):
    '''
    Takes the emb and compute the umap
    '''
    df = emb.copy().reset_index()
    df_for_umap = df[["Z" + str(i) for i in range(emb_size)]]
    Scaled_emb = StandardScaler().fit_transform(df_for_umap)
    reducer = umap.UMAP(random_state=40)
    emb_umap = reducer.fit_transform(Scaled_emb)
    emb_umap = pd.DataFrame(emb_umap, columns=["Z0", "Z1"])

    meta_data = df[[c for c in df.columns if c not in ["Z" + str(i) for i in range(emb_size)]]]

    return emb_umap.join(meta_data)




