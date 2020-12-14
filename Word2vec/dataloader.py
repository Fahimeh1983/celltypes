from torch.utils.data import Dataset
import torch
import numpy as np


class EmitterReceiverDataset(Dataset):
    def __init__(self, tuples_list, wd_2_idx):
        self.context = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in tuples_list]))
        self.target1 = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in tuples_list]))
        self.target2 = torch.from_numpy(np.array([wd_2_idx[i[2]] for i in tuples_list]))
        self.n_samples = len(tuples_list)

    def __getitem__(self, index):
        sample = self.context[index], self.target1[index], self.target2[index]
        # sample = self.context[index], self.target[index]
        return sample

    def __len__(self):
        return self.n_samples


class EmitterReceiverDataset_debug(Dataset):
    def __init__(self, tuples_list, wd_2_idx):
        self.context = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in tuples_list]))
        self.target = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in tuples_list]))
        self.n_samples = len(tuples_list)

    def __getitem__(self, index):
        sample = (self.context[index], self.target[index])
        return sample

    def __len__(self):
        return self.n_samples


class SignedEmitterReceiverDataset(Dataset):
    def __init__(self, triples_list, wd_2_idx):
        self.context = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in triples_list]))
        self.target = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in triples_list]))
        self.edgetype = torch.from_numpy(np.array([i[2] for i in triples_list]))

        self.n_samples = len(triples_list)

    def __getitem__(self, index):
        sample = (self.context[index], self.target[index], self.edgetype[index])
        return sample

    def __len__(self):
        return self.n_samples


class EmitterReceiverNegativeDataset_debug(Dataset):
    def __init__(self, tuples_list, wd_2_idx, n_pos, n_neg):
        self.word = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in tuples_list]))
        self.pos_words = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in tuples_list]))
        self.n_samples = len(tuples_list)

    def __getitem__(self, index):
        sample = (self.context[index], self.target[index])
        return sample

    def __len__(self):
        return self.n_samples


class WalkDataset(Dataset):
    def __init__(self, word_context_tuples_list, wd_2_idx):
        self.target = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in word_context_tuples_list]))
        self.context = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in word_context_tuples_list]))
        self.n_samples = len(word_context_tuples_list)

    def __getitem__(self, index):
        sample = self.target[index], self.context[index]
        return sample

    def __len__(self):
        return self.n_samples


class MCBOW_WalkDataset(Dataset):
    def __init__(self, word_context_tuples_list, wd_2_idx):
        self.target = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in word_context_tuples_list]))
        self.context_list = []
        for l in [i[1] for i in word_context_tuples_list]:
            self.context_list.append([wd_2_idx[j] for j in l])

        self.context_list = torch.from_numpy(np.array(self.context_list))
        self.n_samples = len(word_context_tuples_list)

    def __getitem__(self, index):
        sample = self.target[index], self.context_list[index]
        return sample

    def __len__(self):
        return self.n_samples


class EmitterReceiverNegativeDataset_debug(Dataset):
    def __init__(self, tuples_list, wd_2_idx, n_pos, n_neg):

        words = []
        pos_examples = []
        neg_examples = []

        for exam in tuples_list:
            word = wd_2_idx[exam[0]]
            words.append(word)

            pos_example = tuple()
            for i in range(1, n_pos + 1):
                pos_example = pos_example + (wd_2_idx[exam[i]],)
            pos_examples.append(pos_example)


            neg_example = tuple()
            for i in range(n_pos + 1, n_neg + n_pos + 1):
                neg_example = neg_example + (wd_2_idx[exam[i]],)
            neg_examples.append(neg_example)

        self.words = torch.from_numpy(np.array(words))
        self.pos_examples = torch.from_numpy(np.array(pos_examples))
        self.neg_examples = torch.from_numpy(np.array(neg_examples))
        self.n_samples = len(tuples_list)

    def __getitem__(self, index):
        sample = (self.words[index], self.pos_examples[index], self.neg_examples[index])
        return sample

    def __len__(self):
        return self.n_samples