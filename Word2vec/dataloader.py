from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class WalkDataset(Dataset):
    def __init__(self, word_context_tuples_list, wd_2_idx, transform=None):
        self.target = torch.from_numpy(np.array([wd_2_idx[i[0]] for i in word_context_tuples_list]))
        self.context = torch.from_numpy(np.array([wd_2_idx[i[1]] for i in word_context_tuples_list]))
        self.n_samples = len(word_context_tuples_list)
        self.transform = transform

    def __getitem__(self, index):
        sample = self.target[index], self.context[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        target, context = sample
        return torch.from_numpy(np.array(target)), torch.from_numpy(np.array(context))