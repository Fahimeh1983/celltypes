import torch
import os
import time
import itertools
import torch.nn as nn
import pandas as pd
import numpy as np
from cell.graph_utils import *
from cell.utils import *
from cell.analysis import *
from cell.plot_utils import *
from cell.Word2vec.dataloader import *
from cell.Word2vec.prepare_vocab import *
from cell.Word2vec.wv import *
# from stellargraph import StellarGraph
from torch.nn import functional as F
# from stellargraph.data import BiasedRandomWalk
# import cell.BiasedDirectedWeightedWalk as BDWW
# from stellargraph import StellarDiGraph
# from IPython.display import Image
import matplotlib.pylab as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConcatDataset(torch.utils.data.Dataset):
    '''
    Concatenate datasets of multiple arms
    '''
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def build_data_loader(datasets, batch_size, shuffle=True, drop_last=True, num_workers=0):
    '''
    Args:
        datasets: a dictionary, each key is one arm and values are datasets for that arm
        batch_size: batchsize
        shuffle: True or False for shuffeling the dataset
        drop_last:
        num_workers:
    Returns:
    dataloader
    '''
    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(*[datasets[k][0] for k in datasets.keys()]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return {k: i for i, k in enumerate(datasets.keys())}, data_loader


class EmitterReceiverCoupled(nn.Module):
    """
    """
    def __init__(self, n_nodes, embedding_size, n_arms):
        '''
        Args:
            n_nodes: total number of nodes in the graph
            embedding_size: dimension of the latent space
            n_arms: number of arms
        '''
        super(EmitterReceiverCoupled, self).__init__()
        self.n_nodes = n_nodes
        self.embedding_size = embedding_size
        self.n_arms = n_arms

        self.embeddings = nn.ModuleList([nn.Embedding(n_nodes[i], embedding_size) for i in range(n_arms)])
        self.linear = nn.ModuleList([nn.Linear(embedding_size,  n_nodes[i], bias=False) for i in range(n_arms)])
        self.sigmoid = nn.Sigmoid()

    def encoder(self, first_node, second_node, arm):
        '''
        Encoder is just a look up for first and second node embeddings. If we have a walk from i to j and then k
        one arm receives (j, i) and the other arm will receive (j, k) as example. The first node in both arms is j and
        the second node is i for the first arm and is k for the second arm
        Args:
            first_node: the index of node in the middle of a walk of length 3. In the example above j is the first node
            second_node: the index of node before or the node after the middle node of a walk of length 3. In the example
             above i is the second node for the first arm and k is the second node for the other arm
            arm: arm's key
        Returns:
        a torch tensor of size (batchsize, 2, embedding size) which is the embedding of first and second node for each
        batch example
        '''
        batch_size = first_node.shape[0]
        first_second_embeddings = [self.embeddings[arm](i) for i in [first_node, second_node]]
        first_second_embeddings = torch.stack(first_second_embeddings, dim=1)
        first_second_embeddings = first_second_embeddings.reshape(batch_size, 2, self.embedding_size)
        return first_second_embeddings

    def decoder(self, first_second_embeddings, arm):
        '''
        Takes the embedding of the first and second node, and pass only the first node embedding from a linear and then
        sigmoid layer. In the example above, j is the node in both arms that we take and pass through the linear and
        sigmoid
        Args:
            first_second_embeddings: embedding of the first and second node obtained from encoder
            arm: arm's key
        Returns:
        a torch tensor of size (batch_size, 1, n_nodes)
        '''
        first_embedding = torch.unbind(first_second_embeddings, dim=1)[0]
        out = self.linear[arm](first_embedding)
        out = self.sigmoid(out)
        return out

    def forward(self, first_node, second_node):
        first_second_node_embeddings = [None] * self.n_arms
        output = [None] * self.n_arms

        for arm in range(self.n_arms):
            first_second_embeddings = self.encoder(first_node[arm], second_node[arm], arm)
            first_second_node_embeddings[arm] = first_second_embeddings
            output[arm] = self.decoder(first_second_embeddings, arm)

        first_second_node_embeddings[1] = torch.flip(first_second_node_embeddings[1], [1])

        return first_second_node_embeddings, output


def loss_emitter_receiver_independent(first_second_node_embeddings, batch_size):
    '''
    gets the first and second node embeddings that were obtained from encoder (without passing them to decoder) and
    compute the distance between the first node of one arm and second node of the other arm. In the example above we
    need the distance between j of the second arm and i of the first arm and the distance beween j of the first arm and
    k of secind arm. These distances should be minimized
    Args:
        first_second_node_embeddings: embedding of the first and second nodes that are obtained directly from embedding
        layer. These coordinates are not passed through decoder
        batch_size:
    Returns:
    '''
    dist_squared = torch.norm(first_second_node_embeddings[0] - first_second_node_embeddings[1], dim=2) ** 2
    loss = torch.mean(dist_squared)
    return loss

def loss_AE_independent(output, n_arms, n_nodes, first_node):
    '''
    Take the output which is obtaned from the decoder, this is basically coordinate of j and we want to predict j again
    like and autoencoder using BCE loss
    Args:
        output: output from decoder
        n_arms: number of arms
        n_nodes: number of total nodes in the graph
        first_node: index of first node(node in the middle of a walk), in the example above, the index of node j
    Returns:
    '''

    bce_loss = [None] * n_arms
    for arm in range(n_arms):
        target = (first_node[arm] == torch.arange(n_nodes).reshape(1,n_nodes).to(device)).float()
        loss = nn.BCELoss()
        bce_loss[arm] = loss(output[arm], target)
        # ce_loss[arm] = loss(output[arm], first_node[arm].view(-1))

    return sum(bce_loss)


def min_var_loss(model, n_arms):
    '''
    Compute the variation of embeddings in all direction and take the min
    Args:
        model: model
        n_arms: number of arms
    Returns:
    '''
    m_v_loss = [None] * n_arms
    for arm in range(n_arms):
        zj = model.embeddings[arm].weight
        u, vars_j_, v = torch.svd(zj - torch.mean(zj, dim=0), compute_uv=True)
        m_v_loss[arm] = torch.sqrt(torch.min(vars_j_))
    return min(m_v_loss)


def total_loss(first_second_node_embeddings, batch_size, model, n_arms, output, n_nodes, first_node, lamda):
    '''
    Adding AE loss and the distance loss
    Args:
        first_second_node_embeddings: embedding of the first and second node
        batch_size: batch size
        model: model
        n_arms: number of arms
        output: the output of decoder
        n_nodes: number of total nodes in the graph
        first_node: index of first node (node in the middle of a walk of length 3 for example)
        lamda: the coupling between AE loss and distance loss
    Returns:
    '''
    AE_loss = loss_AE_independent(output, n_arms, n_nodes, first_node)
    mvl = min_var_loss(model, n_arms)
    if torch.isnan(mvl):
        epsilon = 0.001
    else:
        epsilon = 0.
    distance_loss = loss_emitter_receiver_independent(first_second_node_embeddings, batch_size) / (mvl + epsilon)
    return distance_loss + lamda * AE_loss



# Reading the walks for the graph

# p = 1
# q = 1
# N = 1
padding = False
# length = 10000
# roi = "VISp"
# walk_filename = "walk_node21_32_removed.csv"
# project_name = "NPP_GNN_project"
# layer_class = "single_layer"
# layer = "base_unnormalized_allcombined"
# walk_type= "Directed_Weighted_node2vec"
#
# walk_dir = utils.get_walk_dir(roi,
#                               project_name,
#                               N,
#                               length,
#                               p,
#                               q,
#                               layer_class,
#                               layer,
#                               walk_type)
#
# corpus = utils.read_list_of_lists_from_csv(os.path.join(walk_dir, walk_filename))
# vocabulary = prepare_vocab.get_vocabulary(corpus)
#
# print(f'lenght of vocabulary: {len(vocabulary)}')

# word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=False)
# index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=False)

path = "/Users/fahimehb/Documents/NPP_GNN_project/dat/"
walks = utils.read_list_of_lists_from_csv("./walk_node21_32_removed.csv")

# walks = read_list_of_lists_from_csv( path +
#     "/walk_node21_32_removed.csv")

vocabulary = get_vocabulary(walks)
word_2_index = get_word2idx(vocabulary, padding=padding)
index_2_word = get_idx2word(vocabulary, padding=padding)


# Run the code with different values for the window, lambda and embedding size
for w in [1]:
    for e in [5]:
        for l in [1]:
            window = w
            batch_size = 2000
            embedding_size = e
            learning_rate = 0.0001
            n_epochs = 200
            n_arms = 2
            lamda = l

            # receiver_tuples, emitter_tuples = prepare_vocab.emitter_receiver_tuples(corpus, window=window)
            receiver_tuples, emitter_tuples = emitter_receiver_tuples(walks, window=window)
            if padding:
                n_nodes = len(vocabulary) + 1
            else:
                n_nodes = len(vocabulary)

            datasets = {}

            datasets['E'] = []
            emitter_dataset = EmitterReceiverDataset_debug(emitter_tuples, word_2_index)
            datasets['E'].append(emitter_dataset)
            datasets['E'].append(n_nodes)

            datasets['R'] = []
            receiver_dataset = EmitterReceiverDataset_debug(receiver_tuples,
                                                            word_2_index)
            datasets['R'].append(receiver_dataset)
            datasets['R'].append(n_nodes)


            arm_keys, data_loader = build_data_loader(datasets, batch_size=batch_size, shuffle=False)


            model = EmitterReceiverCoupled(embedding_size=embedding_size,
                                     n_nodes=[v[1] for (k, v) in datasets.items()],
                                     n_arms=n_arms).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            training_loss = []

            for epoch in range(n_epochs):
                losses = []
                embs = []
                t0 = time.time()
                for batch_idx, all_data in enumerate(data_loader):
                    first_node = [data[0].to(device) for data in all_data]
                    second_node = [data[1].to(device) for data in all_data]
                    first_node = [torch.reshape(first_node[i], (batch_size, 1)) for i in range(len(first_node))]
                    second_node = [torch.reshape(second_node[i], (batch_size, 1)) for i in range(len(second_node))]
                    optimizer.zero_grad()
                    first_second_node_embeddings, output = model(first_node, second_node)
                    loss = total_loss(first_second_node_embeddings, batch_size, model, n_arms, output, n_nodes, first_node, lamda)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                training_loss.append(np.mean(losses))
                print(f'epoch: {epoch + 1}/{n_epochs}, loss:{np.mean(losses):.4f}')


            R = model.embeddings[0].weight.cpu().detach().numpy()
            R = pd.DataFrame(R, columns=["Z"+str(i) for i in range(embedding_size)], index=index_2_word.values())
            R.index = R.index.astype('str')

            E = model.embeddings[1].weight.cpu().detach().numpy()
            E = pd.DataFrame(E, columns=["Z"+str(i) for i in range(embedding_size)], index=index_2_word.values())
            E.index = E.index.astype('str')

            output_filename = "AE_NPP_BCE_lambda"+ str(l) + "_R_w" + str(window) \
                              + "_bs" + str(batch_size) + "_" + str(
                embedding_size) + "d.csv"
            R.to_csv(path + '/' + "test_R.csv")

            output_filename = "AE_NPP_BCE_lambda"+ str(l) + "_E_w" + str(window) \
                              + "_bs" + str(batch_size) + "_" + \
                              str(embedding_size) + "d.csv"
            E.to_csv(path + "/" + "test_E.csv")

            print("finished w:", w, "embedding size:", e)