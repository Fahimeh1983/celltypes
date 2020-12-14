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
    def __init__(self, n_nodes, n_sign, embedding_size, n_arms):
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
        self.n_sign = n_sign
        # self.l1_size = 5 # adding another layer to decoder if needed

        # encode, an embedding layer followed by a batch normalization
        self.embeddings = nn.ModuleList(
            [nn.Embedding(self.n_nodes, self.embedding_size) for i in range(n_arms)])

        # self.encode_l1 = nn.ModuleList(
        #     [nn.Linear(self.l1_size, self.embedding_size, bias=True) for i in range(n_arms)])

        self.encode_BN1 = nn.ModuleList(
            [nn.BatchNorm1d(self.embedding_size, eps=1e-10, momentum=0.1, affine=False) for i in range(n_arms)])

        # decoder, two linear layers and no batch normalization
        self.decode1_l1 = nn.ModuleList(
            [nn.Linear(self.embedding_size, self.n_nodes, bias=True) for i in range(n_arms)])

        self.decode2_l1 = nn.ModuleList(
            [nn.Linear(2 * self.embedding_size, self.n_sign, bias=True) for i in range(n_arms)])

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

    def decoder1(self, first_second_embeddings, arm):
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

        #TODO
        first_embedding = torch.unbind(first_second_embeddings, dim=1)[1] #Changed it to make it like word2vec
        out = self.decode1_l1[arm](first_embedding)
        # out = self.tanh(out)
        # out = self.decode_l2[arm](out)
        out = self.sigmoid(out)
        return out


    def decoder2(self, first_second_embeddings, arm):
        '''
        get the first and second embedding coordinates and predict the sign. For this I concatenate the embedding of
        the first and second nodes and feed it to the decoder

        Args:
            first_second_embeddings: embedding of the first and second node obtained from encoder
            arm: arm's key

        Returns:
        a torch tensor of size (batch_size, 1, n_nodes)
        '''

        #TODO
        long_first_second_embedding = first_second_embeddings.reshape(batch_size, 2 * embedding_size)
        out = self.decode2_l1[arm](long_first_second_embedding)
        out = self.sigmoid(out)
        return out

    def forward(self, first_node, second_node, index_2_word_tensor):
        '''
        Takes the first nodes, second nodes indices as well as all the nodes indices of the graph and pass them through
        encoder and decoder
        '''

        first_second_node_embeddings = [None] * self.n_arms #place holder for first and second nodes coordinates
        output1 = [None] * self.n_arms #place holder for the output of decoder1
        output2 = [None] * self.n_arms #place holder for the output of decoder2
        all_node_emb = [None] * self.n_arms # place holder for coordinates of all the nodes of the graph

        for arm in range(self.n_arms):
            all_emb, first_second_embeddings = self.encoder(first_node[arm], second_node[arm], index_2_word_tensor, arm)
            first_second_node_embeddings[arm] = first_second_embeddings
            output1[arm] = self.decoder1(first_second_node_embeddings[arm], arm)
            output2[arm] = self.decoder2(first_second_node_embeddings[arm], arm)
            all_node_emb[arm] = all_emb

        first_second_node_embeddings[1] = torch.flip(first_second_node_embeddings[1], [1])

        return all_node_emb, first_second_node_embeddings, output1, output2



def loss_WV(output1, n_arms, n_nodes, first_node):
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
        bce_loss[arm] = loss(output1[arm], target)

    return sum(bce_loss)


def loss_sign(output2, n_arms, n_sign, edge_type):
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
        target = (edge_type[arm] == torch.arange(n_sign).reshape(1, n_sign).to(device)).float()
        loss = nn.BCELoss()
        bce_loss[arm] = loss(output2[arm], target)

    return sum(bce_loss)





def total_loss(n_arms, n_nodes, n_sign, first_node, edge_type, output1, output2):
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
    WV_loss = loss_WV(output1, n_arms, n_nodes, first_node)
    sign_loss = loss_sign(output2, n_arms, n_sign, edge_type)
    return  WV_loss, sign_loss, WV_loss + sign_loss


##############################################
################ Training  ###################
##############################################

padding = False

path = "/Users/fahimehb/Documents/NPP_GNN_project/dat/run_results"
walks = read_list_of_lists_from_csv("/Users/fahimehb/Documents/NPP_GNN_project/dat/all_test_walks/Dynamic_neighbor_walk_npp_version1_35percentile_100_100.csv")
# walks = read_list_of_lists_from_csv("/Users/fahimehb/Documents/GNN/dat/walks/jsd/N_1_l_1000_p_1_q_1/walk_row_normal_new_npp_adj.csv")


combined_edges = pd.read_csv("/Users/fahimehb/Documents/NPP_GNN_project/dat/Signed_edges_version1_50percentile.csv")
combined_edges['source'] = combined_edges['source'].astype(str)
combined_edges['target'] = combined_edges['target'].astype(str)
signed_edges = {}

for idx, row in combined_edges.iterrows():
    edge_prob = []
    edge_prob.append(row['norm_weight_I'])
    edge_prob.append(row['norm_weight_E'])
    edge_prob.append(row['norm_weight_Q'])
    signed_edges[(row['source'], row['target'])] = edge_prob


vocabulary = get_vocabulary(walks)
word_2_index = get_word2idx(vocabulary, padding=padding)
index_2_word = get_idx2word(vocabulary, padding=padding)


# Run the code with different values for the window, lambda and embedding size
for w in [1]: # window size
    for e in [2]: # embedding_size
        for l in [0.99]: # lambda in the loss function
            window = w
            batch_size = 2000
            embedding_size = e
            learning_rate = 0.00001
            n_epochs = 3000
            n_arms = 2
            lamda = l
            n_sign = 3

            # Generating input examples for both arms from the walks
            receiver_triples, emitter_triples = emitter_receiver_edgetype_triples(walks, 1, signed_edges, n_sign)

            # shuffle all the examples
            temp = list(zip(emitter_triples, receiver_triples))
            random.shuffle(temp)
            emitter_triples, receiver_triples = zip(*temp)

            if padding:
                n_nodes = len(vocabulary) + 1
            else:
                n_nodes = len(vocabulary)

            # Create data loader
            datasets = {}
            datasets['E'] = []
            emitter_dataset = SignedEmitterReceiverDataset(emitter_triples, word_2_index)
            datasets['E'].append(emitter_dataset)
            datasets['E'].append(n_nodes)

            datasets['R'] = []
            receiver_dataset = SignedEmitterReceiverDataset(receiver_triples, word_2_index)
            datasets['R'].append(receiver_dataset)
            datasets['R'].append(n_nodes)


            arm_keys, data_loader = build_data_loader(datasets, batch_size=batch_size, shuffle=False)


            # model
            model = EmitterReceiverCoupled(embedding_size=embedding_size,
                                           n_nodes=n_nodes,
                                           n_sign=n_sign,
                                           n_arms=n_arms).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            training_loss = []
            wv_loss = []
            sign_loss = []

            # training
            for epoch in range(n_epochs):
                losses = []

                t0 = time.time()
                for batch_idx, all_data in enumerate(data_loader):

                    first_node = [data[0].to(device) for data in all_data]
                    second_node = [data[1].to(device) for data in all_data]
                    edge_type = [data[2].to(device) for data in all_data]

                    first_node = [torch.reshape(first_node[i], (batch_size, 1)) for i in range(len(first_node))]
                    second_node = [torch.reshape(second_node[i], (batch_size, 1)) for i in range(len(second_node))]
                    edge_type = [torch.reshape(edge_type[i], (batch_size, 1)) for i in range(len(edge_type))]

                    optimizer.zero_grad()
                    all_node_emb, first_second_node_embeddings, output1, output2 = model(
                        first_node, second_node, torch.tensor([i for i in index_2_word.keys()]).to(device)
                    )

                    wv_l, sign_l, loss = total_loss(n_arms, n_nodes, n_sign, first_node, edge_type, output1, output2)

                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                training_loss.append(np.mean(losses))
                wv_loss.append(wv_l.item())
                sign_loss.append(sign_l.item())

                #Saving the outputs every 100 epochs
                print(f'epoch: {epoch + 1}/{n_epochs},'
                      f'loss:{np.mean(losses):.4f},'
                      f'WV:{wv_l:.4f}',
                      f'sign:{sign_l:.4f}')

                if ((epoch % 100 == 0)):
                    R = all_node_emb[0].cpu().detach().numpy()
                    R = pd.DataFrame(R, columns=["Z" + str(i) for i in range(embedding_size)],
                                     index=index_2_word.values())
                    R.index = R.index.astype('str')

                    E = all_node_emb[1].cpu().detach().numpy()
                    E = pd.DataFrame(E, columns=["Z" + str(  i) for i in range(embedding_size)],
                                     index=index_2_word.values())
                    E.index = E.index.astype('str')

                    prefix = "signed_npp_version1_30percentile_run1"
                    output_filename = prefix + "_" + str(epoch) + "_R_w" + str(window) + "_" + \
                                      str(embedding_size) + "d.csv"
                    R.to_csv(path + '/' + output_filename)

                    output_filename = prefix + "_" + str(epoch) + "_E_w" + str(window) + "_" + \
                                      str(embedding_size) + "d.csv"
                    E.to_csv(path + "/" + output_filename)

                    output_filename = prefix + "_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, training_loss)

                    output_filename = prefix + "_WV_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, wv_loss)

                    output_filename = prefix + "_sign_loss.csv"
                    utils.write_list_to_csv(path + '/' + output_filename, sign_loss)

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

            output_filename = prefix + "_WV_loss.csv"
            utils.write_list_to_csv(path + '/' + output_filename, wv_loss)

            output_filename = prefix + "sign.csv"
            utils.write_list_to_csv(path + '/' + output_filename, sign_loss)

            print("finished w:", w, "embedding size:", e)
