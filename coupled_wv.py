import time
import torch
import torch.nn as nn
from cell.Word2vec import prepare_vocab, dataloader, wv
from torch.nn import functional as F
import numpy as np

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


def build_data_loader(datasets, batch_size, shuffle=True, drop_last=True, num_workers=0):
    data_loader = torch.utils.data.DataLoader(
        ConcatDataset(*[datasets[k][0] for k in datasets.keys()]),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return {k: i for i, k in enumerate(datasets.keys())}, data_loader

def return_mcbow_dataloader(walks):

    vocabulary = prepare_vocab.get_vocabulary(walks)

    print(f'lenght of vocabulary: {len(vocabulary)}')

    word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=True)
    index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=True)

    context_tuple_list = prepare_vocab.MCBOW_get_word_context_tuples(walks, window=2)

    dataset = dataloader.MCBOW_WalkDataset(context_tuple_list, word_2_index)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2000,
                                              shuffle=True,
                                              num_workers=0)
    return data_loader, index_2_word, word_2_index, vocabulary

def return_basic_wv_dataloader(walks):

    vocabulary = prepare_vocab.get_vocabulary(walks)

    print(f'lenght of vocabulary: {len(vocabulary)}')

    word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=True)
    index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=True)

    context_tuple_list = prepare_vocab.get_word_context_tuples(walks, window=2)

    dataset = dataloader.WalkDataset(context_tuple_list, word_2_index)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2000,
                                              shuffle=True,
                                              num_workers=0)
    return data_loader, index_2_word, word_2_index, vocabulary


def return_dataloader_coupled_debug(corpus, vocab_size):
    vocabulary = prepare_vocab.get_vocabulary(corpus)
    print(f'lenght of vocabulary: {len(vocabulary)}')

    word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=True)
    index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=True)
    context_tuple_list = prepare_vocab.get_word_context_tuples(corpus, window=2)

    emitter_v_size = vocab_size + 1
    datasets = {}

    datasets['E'] = []
    dataset = dataloader.WalkDataset(context_tuple_list, word_2_index)
    datasets['E'].append(dataset)
    datasets['E'].append(emitter_v_size)

    datasets['R'] = []
    datasets['R'].append(dataset)
    datasets['R'].append(emitter_v_size)

    arm_keys, data_loader = build_data_loader(datasets, batch_size=2000, shuffle=False)
    return arm_keys, data_loader, index_2_word, word_2_index, vocabulary

def return_dataloader_coupled(corpus, padding=True, window=2, shuffle=True, batch_size=2000):

    vocabulary = prepare_vocab.get_vocabulary(corpus)
    word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=padding)
    index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=padding)
    receiver_tuples, emitter_tuples = prepare_vocab.emitter_receiver_tuples(corpus, window=window)

    if padding:
        vocab_size = len(vocabulary) + 1
    else:
        vocab_size = len(vocabulary)

    datasets = {}

    datasets['E'] = []
    emitter_dataset = dataloader.EmitterReceiverDataset(emitter_tuples, word_2_index)
    datasets['E'].append(emitter_dataset)
    datasets['E'].append(vocab_size)

    datasets['R'] = []
    receiver_dataset = dataloader.EmitterReceiverDataset(receiver_tuples, word_2_index)
    datasets['R'].append(receiver_dataset)
    datasets['R'].append(vocab_size)

    arm_keys, data_loader = build_data_loader(datasets, batch_size=batch_size, shuffle=shuffle)
    return arm_keys, data_loader, index_2_word, word_2_index, vocabulary


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
        return self.embeddings[arm](context_word)

    def decoder(self, context_word_embedding_of_the_other_arm, arm):
        return self.linear[arm](context_word_embedding_of_the_other_arm)

    def forward(self, context_word):
        predictions = [None] * self.n_arm
        context_word_embedding = [None] * self.n_arm

        for arm in range(self.n_arm):
            word_embedding = self.encoder(context_word[arm], arm)
            context_word_embedding[arm] = word_embedding

        for arm in range(self.n_arm):
            # which_arm = -1 * arm + 1
            which_arm = arm
            predictions[arm] = self.decoder(context_word_embedding[which_arm], arm)

        return predictions


def loss_emitter_receiver(prediction, target, n_arm, vocab_size, batch_size, arm_keys):
    loss_indep = [None] * n_arm

    for arm, (k, v) in enumerate(arm_keys.items()):
        prediction[arm] = torch.reshape(prediction[arm], (batch_size, vocab_size))
        loss_indep[arm] = F.cross_entropy(prediction[arm], target[arm])

    loss = sum(loss_indep)

    return loss


def run_coupled(data_loader, vocab_size, embedding_size, learning_rate, n_epochs, n_arm, batch_size, device, arm_keys):
    vocab_size = vocab_size
    embedding_size = embedding_size
    learning_rate = learning_rate
    n_epochs = n_epochs
    n_arm = n_arm
    batch_size = batch_size

    model = EmitterReceiver_Word2Vec(embedding_size=embedding_size,
                                    vocab_size=[vocab_size, vocab_size],
                                     n_arm=n_arm).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loss = []

    for epoch in range(n_epochs):
        losses = []
        t0 = time.time()
        for batch_idx, all_data in enumerate(data_loader):
            target_data = [data[1].to(device) for data in all_data]
            context_data = [data[0].to(device) for data in all_data]
            context_data = [torch.reshape(context_data[i], (batch_size, 1)) for i in range(len(context_data))]
            optimizer.zero_grad()
            predict = model(context_data)
            loss = loss_emitter_receiver(predict, target_data, n_arm, vocab_size, batch_size, arm_keys)
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())

        training_loss.append(np.mean(losses))
        if epoch % 9 == 0:
            print(f'epoch: {epoch + 1}/{n_epochs}, loss:{np.mean(losses):.4f}')

    return model, training_loss

def run(vocabulary, embedding_size, learning_rate, n_epochs, data_loader, device):
    vocab_size = len(vocabulary) + 1
    embedding_size = embedding_size
    learning_rate = learning_rate
    n_epochs = n_epochs

    criterion = nn.CrossEntropyLoss()

    model = wv.MCBOW_Word2Vec(embedding_size=embedding_size, vocab_size=vocab_size)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(data_loader)

    training_loss = []

    for epoch in range(n_epochs):
        t0 = time.time()
        losses = []
        for i, (target, context) in enumerate(data_loader):
            target = target.to(device)
            context = context.to(device)
            prediction = model(context)
            loss = criterion(prediction, target)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        t1 = time.time()
        # print('time is %.2f' % (t1 - t0))

        training_loss.append(np.mean(losses))
        if epoch % 9 == 0:
            print(f'epoch: {epoch + 1}/{n_epochs}, loss:{np.mean(losses):.4f}')

    return model, training_loss


def run_basic_wv(vocabulary, embedding_size, learning_rate, n_epochs, data_loader, device):
    vocab_size = len(vocabulary) + 1
    embedding_size = embedding_size
    learning_rate = learning_rate
    n_epochs = n_epochs

    criterion = nn.CrossEntropyLoss()

    model = wv.Word2Vec(embedding_size=embedding_size, vocab_size=vocab_size)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(data_loader)

    training_loss = []

    for epoch in range(n_epochs):
        t0 = time.time()
        losses = []
        for i, (target, context) in enumerate(data_loader):
            target = target.to(device)
            context = context.to(device)
            prediction = model(context)
            loss = criterion(prediction, target)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        t1 = time.time()
        # print('time is %.2f' % (t1 - t0))

        training_loss.append(np.mean(losses))
        if epoch % 9 == 0:
            print(f'epoch: {epoch + 1}/{n_epochs}, loss:{np.mean(losses):.4f}')

    return model, training_loss


class EmitterReceiver_Word2Vec_2arms(nn.Module):
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

    def encoder(self, context_word, arm):
        h1 = self.embeddings[arm](context_word)
        return h1

    def decoder(self, context_word_embedding_of_the_other_arm, arm):
        h2 = self.linear[arm](context_word_embedding_of_the_other_arm)
        return h2

    def forward(self, context_word):
        emb = [None] * self.n_arm
        predictions = [None] * self.n_arm
        context_word_embedding = [None] * self.n_arm

        for arm in range(self.n_arm):
            word_embedding = self.encoder(context_word[arm], arm)
            context_word_embedding[arm] = word_embedding

        for arm in range(self.n_arm):
            which_arm = arm * -1 + 1
            predictions[arm] = self.decoder(context_word_embedding[which_arm], arm)

        return predictions


def loss_emitter_receiver_2arms(predictions, target,
                                n_arm, vocab_size, batch_size):
    loss_indep = [None] * n_arm

    for arm, (k, v) in enumerate(arm_keys.items()):
        predictions[arm] = torch.reshape(predictions[arm], (batch_size, vocab_size))
        loss_indep[arm] = F.cross_entropy(predictions[arm], target[arm])

    loss = sum(loss_indep)

    return loss
