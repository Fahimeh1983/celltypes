#to run:
#python -m trainer --N N --length length --p p --q q --walk_filename walk_filename --roi roi
# --project_name project_name  --layer_class layer_class --layer layer --walk_type walk_type
# --window window --batch_size batch_size --num_workers num_workers --embedding_size embedding_size
# --learning_rate learning_rate --n_epochs n_epochs


import os
import time
import torch
import argparse
import timeit

import numpy as np
import torch.nn as nn

from cell import utils, analysis
from cell.Word2vec import prepare_vocab, dataloader, wv

parser = argparse.ArgumentParser()
parser.add_argument("--IO_files", default="IO_path.csv", type=str, help="IO path")
parser.add_argument("--window", default=None, type=int, help="window size for contex-tuple pair")
parser.add_argument("--batch_size", default=None, type=int, help="batch size")
parser.add_argument("--num_workers", default=1, type=int, help="number of workers for dataloader")
parser.add_argument("--embedding_size", default=None, type=int, help="embedding_size")
parser.add_argument("--learning_rate", default=None, type=float, help="learning_rate")
parser.add_argument("--n_epochs", default=1, type=int, help="n_epochs")
parser.add_argument("--MCBOW", default=None, type=bool, help="If True then all MCBOW classes and vocab "
                                                              "prepartion will be done with padding at index zero")
parser.add_argument("--gpu_device", default=None, type=str, help="name of the gpu. e.g cuda:0")





def main(IO_files, window, batch_size, num_workers, embedding_size, learning_rate, n_epochs, MCBOW, gpu_device):

    #gpu config
    device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')

    pwd = os.path.dirname(os.path.realpath(__file__))
    walk_path, model_path, loss_path = utils.read_list_from_csv(pwd, IO_files)
    #walk_dir = utils.get_walk_dir(roi, project_name, N, length, p, q, layer_class, layer, walk_type)

    #prepare vocabulary
    corpus = utils.read_list_of_lists_from_csv(walk_path)
    vocabulary = prepare_vocab.get_vocabulary(corpus)
    print(f'length of vocabulary: {len(vocabulary)}')
    word_2_index = prepare_vocab.get_word2idx(vocabulary, padding=padding)
    index_2_word = prepare_vocab.get_idx2word(vocabulary, padding=padding)

    #prepare context-word pairs and data loader
    if MCBOW:
        print("Since padding is true we are using CBOW word context tuple generators Also "
              "for dataset generator, MCBOW WalkDataset is being used")
        context_tuple_list = prepare_vocab.MCBOW_get_word_context_tuples(corpus, window=window)
        dataset = dataloader.MCBOW_WalkDataset(context_tuple_list, word_2_index)

    else:
        context_tuple_list = prepare_vocab.get_word_context_tuples(corpus, window=window)
        dataset = dataloader.WalkDataset(context_tuple_list, word_2_index)

    #dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_workers)

    #training loop
    if MCBOW:
        print("Vocab size is len(vocabulary) + 1 because of padding and MCBOW_word2vec is being used")
        vocab_size = len(vocabulary) + 1
        model = wv.MCBOW_Word2Vec(embedding_size=embedding_size, vocab_size=vocab_size)

    else:
        vocab_size = len(vocabulary)
        model = wv.Word2Vec(embedding_size=embedding_size, vocab_size=vocab_size)

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_loss = []

    start_time = timeit.default_timer()

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
        print('time is %.2f' % (t1 - t0))

        training_loss.append(np.mean(losses))
        print(f'epoch: {epoch + 1}/{n_epochs}, loss:{np.mean(losses):.4f}')

    print("Done!")

    #save embedding and loss
    vectors = model.embeddings.weight.detach().cpu().numpy()
    data = analysis.summarize_walk_embedding_results(gensim_dict={"model": vectors},
                                                     index=index_2_word.values(),
                                                     ndim=embedding_size)


    data.to_csv(model_path)

    utils.write_list_to_csv(loss_path, training_loss)

    elapsed = timeit.default_timer() - start_time

    print('-------------------------------')
    print('Training time:', elapsed)
    print('-------------------------------')

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
