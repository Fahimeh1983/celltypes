from torch.utils.data import Dataset
import torch
import numpy as np

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