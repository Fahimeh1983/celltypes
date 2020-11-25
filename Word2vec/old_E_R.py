import warnings; warnings.simplefilter('ignore')

import torch
import os
import time
import itertools
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn import functional as F
import matplotlib.pylab as plt
from cell.graph_utils import *
from cell.utils import *
from cell.analysis import *
from cell.plot_utils import *
from cell.Word2vec.wv import *
from cell.Word2vec.dataloader import *
from cell.Word2vec.prepare_vocab import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################
## Processing data and building data loader ##
##############################################

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
        batch_size: batch size
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

##############################################
########  Model and loss function  ###########
##############################################

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
        self.l1_size = 5 # adding another layer to decoder if needed

        # encode, an embedding layer followed by a batch normalization
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.n_nodes, self.embedding_size) for i in range(n_arms)])

        # self.encode_l1 = nn.ModuleList(
        #     [nn.Linear(self.l1_size, self.embedding_size, bias=True) for i in range(n_arms)])

        self.encode_BN1 = nn.ModuleList(
            [nn.BatchNorm1d(self.embedding_size, eps=1e-10, momentum=0.1, affine=False) for i in range(n_arms)])

        # decoder, two linear layers and no batch normalization
        self.decode_l1 = nn.ModuleList(
            [nn.Linear(self.embedding_size, self.n_nodes, bias=True) for i in range(n_arms)])

        # self.decode_l2 = nn.ModuleList(
        #     [nn.Linear(self.l1_size, self.n_nodes, bias=True) for i in range(n_arms)])

        # non_linearity
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() # if needed for encoder non-linearity

    def encoder(self, first_node, second_node, index_2_word_tensor, arm):
        '''
        Encoder is just a look up for first and second node embeddings. If we have a walk from i to j and then k
        one arm receives (j, i) and the other arm will receive (j, k) as example. The first node in both arms is j and
        the second node is i for the first arm and is k for the second arm

        Args:
            first_node: the index of node in the middle of a walk of length 3. In the example above j is the first node
            second_node: the index of node before or the node after the middle node of a walk of length 3. In the example
             above i is the second node for the first arm and k is the second node for the other arm
            index_2_word_tensor : The indices of all the nodes in the graph provided as a tensor, this is used to read
             the embedding of all nodes
            arm: arm's key

        Returns:
            all_emb: the coordinates of all the nodes of the graph for that arm
            first_second_embeddings: a torch tensor of size (batchsize, 2, embedding size) which is the embedding of
            first and second node for each batch example
        '''

        batch_size = first_node.shape[0]

        # here we are passing the index of first node and second node for all the points in the batch as well as the
        # index of all nodes of the graph
        tmp = []
        if arm == 1:
            for node_index_list in [first_node.reshape(batch_size), second_node.reshape(batch_size),
                                index_2_word_tensor]:
                out = self.embeddings[arm](node_index_list)
                # out = self.encode_l1[arm](out)
                # out = self.tanh(out)
                out = self.encode_BN1[arm](out)
                tmp.append(out)
        else:
            for node_index_list in [first_node.reshape(batch_size), second_node.reshape(batch_size),
                                    index_2_word_tensor]:
                out = self.embeddings[arm](node_index_list)
                # out = self.encode_l1[arm](out)
                # out = self.tanh(out)
                tmp.append(out)


        first = tmp[0] # the coordinates of all the first nodes in the batch
        second = tmp[1] # the coordinates of all the second nodes in the batch
        all_emb = tmp[2] # the coordinates of all the nodes in the graph

        first_second_embeddings = torch.stack((first, second), dim=1)
        first_second_embeddings = first_second_embeddings.reshape(batch_size, 2, self.embedding_size)

        return all_emb, first_second_embeddings

    def decoder(self, first_second_embeddings, arm):
        '''
        Takes the embedding of the first and second node, and pass only the first node embedding from a linear then a
        relu and then another linear and finally sigmoid. In the example above, j is the node in both arms that we take
        and pass through the decoder

        Args:
            first_second_embeddings: embedding of the first and second node obtained from encoder
            arm: arm's key

        Returns:
        a torch tensor of size (batch_size, 1, n_nodes)
        '''

        first_embedding = torch.unbind(first_second_embeddings, dim=1)[0]
        out = self.decode_l1[arm](first_embedding)
        # out = self.tanh(out)
        # out = self.decode_l2[arm](out)
        out = self.sigmoid(out)
        return out

    def forward(self, first_node, second_node, index_2_word_tensor):
        '''
        Takes the first nodes, second nodes indices as well as all the nodes indices of the graph and pass them through
        encoder and decoder
        '''

        first_second_node_embeddings = [None] * self.n_arms #place holder for first and second nodes coordinates
        output = [None] * self.n_arms #place holder for the output of decoder
        all_node_emb = [None] * self.n_arms # place holder for coordinates of all the nodes of the graph

        for arm in range(self.n_arms):
            all_emb, first_second_embeddings = self.encoder(first_node[arm], second_node[arm], index_2_word_tensor, arm)
            first_second_node_embeddings[arm] = first_second_embeddings
            output[arm] = self.decoder(first_second_node_embeddings[arm], arm)
            all_node_emb[arm] = all_emb

        first_second_node_embeddings[1] = torch.flip(first_second_node_embeddings[1], [1])

        return all_node_emb, first_second_node_embeddings, output


def loss_emitter_receiver_independent(first_second_node_embeddings):
    '''
    gets the first and second node embeddings that were obtained from encoder (without passing them to decoder) and
    compute the distance between the first node of one arm and second node of the other arm. In the example above we
    need the distance between j of the second arm and i of the first arm and the distance between j of the first arm and
    k of second arm. These distances should be minimized in the loss function then

    Args:
        first_second_node_embeddings: embedding of the first and second nodes that are obtained directly from embedding
        layers for each arm. These coordinates are not passed through decoder and are obtained from encoder

    Returns:
        loss: distance loss which is the mean of the squared distance between all the first and second nodes in the batch
    '''

    dist_squared = torch.norm(first_second_node_embeddings[0] - first_second_node_embeddings[1], dim=2) ** 2
    loss = torch.mean(dist_squared)
    # loss = torch.mean(torch.unique(dist_squared.reshape(4000)))

    return loss

def loss_AE_independent(output, n_arms, n_nodes, first_node):
    '''
    Take the output which is obtained from the decoder, this is basically coordinate of j and we want to predict j again
    like and autoencoder using BCE loss

    Args:
        output: output from decoder, this is the output of node j coordinates that is passed through decoder
        n_arms: number of arms
        n_nodes: number of total nodes in the graph
        first_node: index of first node(node in the middle of a walk), in the example above, the index of node j
    Returns:
    '''

    bce_loss = [None] * n_arms
    for arm in range(n_arms):
        # here we convert the index of j to its one-hot representation
        target = (first_node[arm] == torch.arange(n_nodes).reshape(1, n_nodes).to(device)).float()
        loss = nn.BCELoss()
        bce_loss[arm] = loss(output[arm], target)
        # ce_loss[arm] = loss(output[arm], first_node[arm].view(-1))

    return sum(bce_loss)


def min_var_loss(first_second_node_embeddings):
    '''
    Compute the variation of embeddings in all direction and take the min

    Args:
        first_second_node_embeddings: embedding of the first and second node from both arms taken from encoder
        embedding_size: embedding dimension

    Returns:
       sqrt of the min of the svd in all direction of embedding space
    '''

    batch_size = first_second_node_embeddings[0].shape[0]
    embedding_size = first_second_node_embeddings[0].shape[2]

    zj = torch.stack((first_second_node_embeddings[0],
                      first_second_node_embeddings[1]),
                     dim=0).reshape(4 * batch_size, embedding_size)

    u, vars_j_, v = torch.svd(zj - torch.mean(zj, dim=0), compute_uv=True)
    m_v_loss = torch.sqrt(torch.min(vars_j_))
    return torch.sqrt(vars_j_), m_v_loss


def total_loss(first_second_node_embeddings, n_arms, output, n_nodes, first_node, lamda):
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
    bothmvl, mvl = min_var_loss(first_second_node_embeddings)
    if torch.isnan(mvl):
        epsilon = 0.001
    else:
        epsilon = 0.
    mean_dist = loss_emitter_receiver_independent(first_second_node_embeddings)
    distance_loss = mean_dist
    return mean_dist, distance_loss, bothmvl, mvl, AE_loss,  lamda * (distance_loss / mvl) + (1 - lamda) * AE_loss


##############################################
################ Training  ###################
##############################################

padding = False

path = "/Users/fahimehb/Documents/NPP_GNN_project/dat/"
walks = read_list_of_lists_from_csv("/Users/fahimehb/Documents/NPP_GNN_project/dat/walk_extended_npp_100_100.csv")
# walks = read_list_of_lists_from_csv("/Users/fahimehb/Documents/GNN/dat/walks/jsd/N_1_l_1000_p_1_q_1/walk_row_normal_new_npp_adj.csv")


vocabulary = get_vocabulary(walks)
word_2_index = get_word2idx(vocabulary, padding=padding)
index_2_word = get_idx2word(vocabulary, padding=padding)


# Run the code with different values for the window, lambda and embedding size
for w in [1]: # window size
    for e in [3]: # embedding_size
        for l in [0.99]: # lambda in the loss function
            window = w
            batch_size = 2000
            embedding_size = e
            learning_rate = 0.001
            n_epochs = 3000
            n_arms = 2
            lamda = l

            # Generating input examples for both arms from the walks
            receiver_tuples, emitter_tuples = emitter_receiver_tuples(walks, window=window)

            # shuffle all the examples
            temp = list(zip(emitter_tuples, receiver_tuples))
            random.shuffle(temp)
            emitter_tuples, receiver_tuples = zip(*temp)


            if padding:
                n_nodes = len(vocabulary) + 1
            else:
                n_nodes = len(vocabulary)

            # Create data loader
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


            # model
            model = EmitterReceiverCoupled(embedding_size=embedding_size,
                                     n_nodes=n_nodes,
                                     n_arms=n_arms).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            training_loss = []
            mean_d = []
            dist_loss = []
            minvar = []
            AEloss = []
            mvl0 =[]
            mvl1 =[]
            # mvl2 = []

            # training
            for epoch in range(n_epochs):
                losses = []

                t0 = time.time()
                for batch_idx, all_data in enumerate(data_loader):
                    first_node = [data[0].to(device) for data in all_data]
                    second_node = [data[1].to(device) for data in all_data]
                    first_node = [torch.reshape(first_node[i], (batch_size, 1)) for i in range(len(first_node))]
                    second_node = [torch.reshape(second_node[i], (batch_size, 1)) for i in range(len(second_node))]
                    optimizer.zero_grad()
                    all_node_emb, first_second_node_embeddings, output = model(first_node, second_node, torch.tensor(
                        [i for i in index_2_word.keys()]).to(device))
                    d, dloss, bothmv, minv, AE, loss = total_loss(
                        first_second_node_embeddings, n_arms, output, n_nodes, first_node, lamda)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                training_loss.append(np.mean(losses))
                AEloss.append(AE.item())
                dist_loss.append(dloss.item())
                minvar.append(minv.item())
                mean_d.append(d.item())
                mvl0.append(bothmv[0].item())
                mvl1.append(bothmv[1].item())
                # mvl2.append(bothmv[2].item())

                #Saving the outputs every 100 epochs
                print(f'epoch: {epoch + 1}/{n_epochs},'
                      f' mean_d:{d:.4f}, '
                      f'dist_loss:{dloss:.4f}, '
                      f'mvl0:{bothmv[0]:.4f}, '
                      f'mvl1:{bothmv[1]:.4f}, '
                      # f'mvl2:{bothmv[2]:.4f}, '
                      f'loss:{np.mean(losses):.4f},'
                      f'AE:{AE:.4f}')

                if ((epoch % 100 == 0) & (epoch >0)):
                    R = all_node_emb[0].cpu().detach().numpy()
                    R = pd.DataFrame(R, columns=["Z" + str(i) for i in range(embedding_size)],
                                     index=index_2_word.values())
                    R.index = R.index.astype('str')

                    E = all_node_emb[1].cpu().detach().numpy()
                    E = pd.DataFrame(E, columns=["Z" + str(i) for i in range(embedding_size)],
                                     index=index_2_word.values())
                    E.index = E.index.astype('str')

                    prefix = "extended_npp_run0"
                    output_filename = prefix + "_" + str(epoch) + "_R_w" + str(window) + "_" + \
                                      str(embedding_size) + "d.csv"
                    R.to_csv(path + '/' + output_filename)

                    output_filename = prefix + "_" + str(epoch) + "_E_w" + str(window) + "_" + \
                                      str(embedding_size) + "d.csv"
                    E.to_csv(path + "/" + output_filename)

                    output_filename = prefix + "_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, training_loss)

                    output_filename = prefix + "_dist_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, dist_loss)

                    output_filename = prefix + "_AE_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, AEloss)

                    output_filename = prefix + "_minvar.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, minvar)

                    output_filename = prefix + "_mean_d.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, mean_d)

                    output_filename = prefix + "_mvl0.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, mvl0)

                    output_filename = prefix + "_mvl1.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, mvl1)
                    #
                    # output_filename = prefix + "_mvl2.csv"
                    # utils.write_list_to_csv(path + '/' + output_filename, mvl2)


                    print("finished w:", w, "embedding size:", e)

            # Saving final output
            all_node_emb, first_second_node_embeddings, output = model(first_node, second_node, torch.tensor(
                [i for i in index_2_word.keys()]).to(device))

            R = all_node_emb[0].cpu().detach().numpy()
            R = pd.DataFrame(R, columns=["Z"+str(i) for i in range(embedding_size)], index=index_2_word.values())
            R.index = R.index.astype('str')

            E = all_node_emb[1].cpu().detach().numpy()
            E = pd.DataFrame(E, columns=["Z"+str(i) for i in range(embedding_size)], index=index_2_word.values())
            E.index = E.index.astype('str')

            output_filename = prefix + "_R_w" + str(window)  + "_" + \
                              str(embedding_size) + "d.csv"
            R.to_csv(path + '/' + output_filename)

            output_filename = prefix + "_E_w" + str(window) + "_" + \
                              str(embedding_size) + "d.csv"
            E.to_csv(path + "/" + output_filename)

            output_filename = prefix + "_loss.csv"
            utils.write_list_to_csv(path + '/' + output_filename, training_loss)

            output_filename = prefix + "_dist_loss.csv"
            utils.write_list_to_csv(path + '/' + output_filename,  dist_loss)

            output_filename = prefix + "_AE_loss.csv"
            utils.write_list_to_csv(path + '/' + output_filename, AEloss)

            output_filename = prefix + "_minvar.csv"
            utils.write_list_to_csv(path + '/' + output_filename, minvar)

            output_filename = prefix + "_mean_d.csv"
            utils.write_list_to_csv(path + '/' + output_filename, mean_d)

            output_filename = prefix + "_mvl0.csv"
            utils.write_list_to_csv(path + '/' + output_filename, mvl0)

            output_filename = prefix + "_mvl1.csv"
            utils.write_list_to_csv(path + '/' + output_filename, mvl1)

            # output_filename = prefix + "_mvl2.csv"
            # utils.write_list_to_csv(path + '/' + output_filename, mvl2)


            print("finished w:", w, "embedding size:", e)
