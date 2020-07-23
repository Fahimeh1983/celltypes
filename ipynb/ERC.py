import torch
import os
import time
import itertools
import torch.nn as nn
from cell import graph_utils, utils
from cell.Word2vec import prepare_vocab, dataloader, wv
from stellargraph import StellarGraph
from torch.nn import functional as F
from stellargraph.data import BiasedRandomWalk
import cell.BiasedDirectedWeightedWalk as BDWW
from stellargraph import StellarDiGraph
from cell import  utils, analysis, plot_utils
from cell.Word2vec import prepare_vocab, dataloader, wv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    
def build_data_loader(datasets, batch_size, shuffle=True, drop_last=True, num_workers=1):
    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(*[datasets[k][0] for k in datasets.keys()]),
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return {k:i for i,k in enumerate(datasets.keys())}, data_loader

class EmitterReceiver_Word2Vec(nn.Module):
    """
    """
    def __init__(self, vocab_size=[93], embedding_size=2, n_arm=1):
        """
        """
        super(EmitterReceiver_Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_arm = n_arm
        
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i],
                                                      embedding_size) 
                                         for i in range(n_arm)])
        
        self.linear = nn.ModuleList([nn.Linear(embedding_size,
                                               vocab_size[i]) 
                                     for i in range(n_arm)])
        
#         self.batch_norm = nn.ModuleList([nn.BatchNorm1d(num_features=embedding_size,
#                                                         eps=1e-10, 
#                                                         momentum=0.1, 
#                                                         affine=False) 
#                                          for i in range(n_arm)])
                        

    def encoder(self, context_word, arm):
        h1 = self.embeddings[arm](context_word)
        node_embeddings = [self.embeddings[arm](torch.tensor(i)) for i 
                           in range(self.vocab_size[arm])]
        return node_embeddings, h1

    def decoder(self, context_word_embedding_of_the_other_arm, arm):
        h2 = self.linear[arm](context_word_embedding_of_the_other_arm)
        return h2

    def forward(self, context_word):
        emb = [None] * self.n_arm
        predictions = [None] * self.n_arm
        context_word_embedding = [None] * self.n_arm
        
        for arm in range(self.n_arm):
            node_embeddings, word_embedding  = self.encoder(context_word[arm], arm)
            emb[arm] = node_embeddings
            context_word_embedding[arm] = word_embedding
            
        for arm in range(self.n_arm):
            which_arm = -1 * arm + 1
#             which_arm = arm
            predictions[arm] = self.decoder(context_word_embedding[which_arm], arm)
            
        return emb, predictions

    
def loss_emitter_receiver(prediction, target, n_arm, vocab_size, batch_size):
 
    loss_indep = [None] * n_arm
    
    for arm, (k, v) in enumerate(arm_keys.items()):
        predict[arm] = torch.reshape(prediction[arm], (batch_size, vocab_size))
        loss_indep[arm] = F.cross_entropy(prediction[arm], target[arm])
                
    loss = sum(loss_indep)

    return loss

length = 10000
p = 1
q = 1
N = 1
batch_size = 2000
walk_filename = "walk_node21_32_removed.csv"
roi = "VISp"
project_name = "NPP_GNN_project"
layer_class = "single_layer"
layer = "base_unnormalized_allcombined"
walk_type= "Directed_Weighted_node2vec"

walk_dir = utils.get_walk_dir(roi,
                              project_name, 
                              N, 
                              length, 
                              p, 
                              q, 
                              layer_class, 
                              layer, 
                              walk_type) 
print(walk_dir)
