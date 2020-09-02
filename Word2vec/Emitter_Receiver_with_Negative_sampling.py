from cell import math_utils
import torch
import torch.nn as nn


def emitter_receiver_negative_tuples(corpus, window, n_neg, negative_prob_table, reversed_negative_prob_table):
    emitter_tuple_list = []
    receiver_tuple_list = []
    for text in corpus:
        for i, word in enumerate(text):
            first_context_word_index = max(0, i - window)
            last_context_word_index = min(i + window + 1, len(text))
            if (i >= window) & (i < len(text) - window):
                emitter_negs = [math_utils.weighted_choice([f for f in negative_prob_table[word].keys()],
                                                [p for p in negative_prob_table[word].values()]) for it in range(n_neg)]

                receiver_negs = [math_utils.weighted_choice([f for f in reversed_negative_prob_table[word].keys()],
                                                 [p for p in reversed_negative_prob_table[word].values()]) for it in
                                 range(n_neg)]

                pos_neg_examples = (word,)
                for j in range(first_context_word_index, i):
                    pos_neg_examples = pos_neg_examples + (text[j],)
                for neg in receiver_negs:
                    pos_neg_examples = pos_neg_examples + (neg,)
                receiver_tuple_list.append(pos_neg_examples)

                pos_neg_examples = (word,)
                for j in range(last_context_word_index - 1, i, -1):
                    pos_neg_examples = pos_neg_examples + (text[j],)
                for neg in emitter_negs:
                    pos_neg_examples = pos_neg_examples + (neg,)
                emitter_tuple_list.append(pos_neg_examples)
    return emitter_tuple_list, receiver_tuple_list


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *data_sets):
        self.datasets = data_sets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def build_data_loader(data_sets, batch_size, shuffle=True, drop_last=True, num_workers=0):
    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(*[data_sets[k][0] for k in data_sets.keys()]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return {k: i for i, k in enumerate(data_sets.keys())}, data_loader






# original_edges = pd.DataFrame([['1', '2', 0.8],
#                       ['1', '3', 0.1],
#                       ['1', '4', 0.05],
#                       ['1', '7', 0.05],
#                       ['2', '6', 0.7],
#                       ['2', '9', 0.3],
#                       ['3', '9', 0.5],
#                       ['3', '7', 0.5],
#                       ['4', '8', 0.95],
#                       ['4', '5', 0.05],
#                       ['5', '1', 1],
#                       ['6', '9', 0.2],
#                       ['6', '10', 0.8],
#                       ['7', '10', 0.15],
#                       ['7', '11', 0.85],
#                       ['8', '5', 0.55],
#                       ['8', '4', 0.45],
#                       ['9', '10', 1],
#                       ['10', '11', 0.6],
#                       ['10', '6', 0.4],
#                       ['11', '8', 0.1],
#                       ['11', '5', 0.9]], columns=['source', 'target', 'weight'])
#
# inverted_edges = pd.DataFrame([['2', '1', 1],
#                       ['3', '1', 1],
#                       ['4', '1', 0.1],
#                       ['7', '1', 0.05/(0.05+0.5)],
#                       ['6', '2', 0.7/(0.7+0.4)],
#                       ['9', '2', 0.3/(0.3+0.5+0.2)],
#                       ['9', '3', 0.5/(0.3+0.5+0.2)],
#                       ['7', '3', 0.5/(0.05+0.5)],
#                       ['8', '4', 0.95/(0.95+0.1)],
#                       ['5', '4', 0.05/(0.05+0.55+0.9)],
#                       ['1', '5', 1],
#                       ['9', '6', 0.2/(0.3+0.5+0.2)],
#                       ['10', '6', 0.8/(0.8+0.15+1)],
#                       ['10', '7', 0.15/(0.8+0.15+1)],
#                       ['11', '7', 0.85/(0.85+0.6)],
#                       ['5', '8', 0.55/(0.05+0.55+0.9)],
#                       ['4', '8', 0.9],
#                       ['10', '9', 1/(0.8+0.15+1)],
#                       ['11', '10', 0.6/(0.85+0.6)],
#                       ['6', '10', 0.4/(0.7+0.4)],
#                       ['8', '11', 0.1/(0.95+0.1)],
#                       ['5', '11', 0.9/(0.05+0.55+0.9)]], columns=['source', 'target', 'weight'])
#
# print(sum(original_edges['weight'] == inverted_edges['weight'])==original_edges.shape[0])
# print(sum(original_edges['source'] == inverted_edges['target'])==original_edges.shape[0])
# print(sum(original_edges['target'] == inverted_edges['source'])==original_edges.shape[0])
#
# edges= original_edges
#
# weight_mat = return_weight_mat_from_edgelist(edges, directed=True)
# weight_mat = weight_mat.loc[[str(i) for i in range(1,12)]][[str(i) for i in range(1,12)]]
# weight_mat
#
# # 1) for each layer first create a nx-Digraph
# nxg = build_nx_graph(source_target_weight=edges, directed=True)
#
# # 2) Create stellar Di graphs
# sdg = StellarDiGraph(nxg)
#
# BDWW.BeginWalk(sdg, begin_checks=True, weighted=True, directed=True)
#
# rw = BDWW.BiasedDirectedRandomWalk(sdg,
#                                    directed=True,
#                                    weighted=True,
#                                    begin_checks=False)
#
# nodes = list(sdg.nodes())
# walks = rw.run(nodes=nodes,
#                length=11,
#                n=100,
#                p=1,
#                q=1,
#                weighted=True,
#                directed=True)
#
# write_list_of_lists_to_csv("/Users/fahimehb/Documents/NPP_GNN_project/dat/10_footbal_weighted-directed_walk.csv", walks)
#
# write_list_of_lists_to_csv("/Users/fahimehb/Documents/NPP_GNN_project/dat/10_inverted_footbal_weighted-directed_walk.csv", walks)