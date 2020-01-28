#!/usr/bin/env python
__author__ = 'Fahimeh Baftizadeh'
__copyright__ = 'Copyright 2020, Cell type'
__email__ = 'fahimeh.baftizadeh@gmail.com'

import numpy as np
import pandas as pd
import scipy

def Balanced_sampling(cl, n_sample, groupby, replace=False):
    """
    take n sample from each group

    parameters
    ----------
    cl: a datafram with sample ids and cluster labels

    n_sample: number of sample from each group

    groupby : groupby variable


    return
    ----------
    a datafram  of sampled data

    """

    fn = lambda obj: obj.loc[np.random.choice(obj.index, n_sample, replace), :]
    sample_data = cl.groupby(groupby, as_index=False).apply(fn)

    return sample_data


def Get_Median_per_Type(df, ref_cl, groupby_col, merge_col, normalize = True):
    '''
    Group by the type and then find the median per type

    parameters
    ----------
    df: a data frame which has a column as id_col and the rest are the
    gene expression values

    ref_cl : a data frame which has a column as id_col and the rest are
    the label for which we will do the groupby

    groupby_col: is the name of the column of ref_cl for which we are
    grouping the data

    merge_col : is the common column between df and ref_cl for which we are
    merging the data. it should be a string

    return
    ----------
    dict :  a dictionary which has keys as the groupby labels and the values
    are the median gene expression per label

    '''

    df = df.merge(ref_cl, on=merge_col)
    groups = df.groupby(groupby_col)
    output_dict = {}

    for g, data in groups:
        data = data.drop([merge_col, groupby_col], axis=1)
        output_dict[g] = data.sum(axis=0)  # Find the colsum for each group
        if normalize:
            output_dict[g] = output_dict[g] / sum(output_dict[g])  # normalize

    return output_dict


def Compute_JSD_for_All_Keys(mydict, sorted_labels, log_base=2, normalize_to_one=False):
    '''
    Take a dictionary and find JSD between each pair

    parameters
    ----------
    mydict: a dictionary that the values for each key is normalized

    log_base: base of log for JSD calculation

    normalize_to_one: If set to True, then the total JSD will be normalized
    to the max value

    sorted_labels: the order of column and rows for jsd output

    return
    ----------
    jsd :  a data frame with the JSD values

    '''

    jsd = pd.DataFrame(index=sorted_labels, columns=sorted_labels)

    for t1 in mydict.keys():
        for t2 in mydict.keys():
             jsd_value = scipy.spatial.distance.jensenshannon(mydict[t1], mydict[t2], log_base)
             jsd[t1][t2] = pow(jsd_value, 2)
    jsd = jsd[jsd.columns].astype(float)

    if normalize_to_one:
        jsd = jsd / max(np.max(jsd))

    jsd = jsd[sorted_labels][sorted_labels]

    return jsd

