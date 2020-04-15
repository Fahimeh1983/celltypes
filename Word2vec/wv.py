import torch.nn  as  nn
import torch

class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, context_word):
        out = self.embeddings(context_word)
        out = self.linear(out)
        return out


class MCBOW_Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(MCBOW_Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.linear = nn.Linear(embedding_size, vocab_size)

        # context.size : (batch_size, window * 2)
        # self.embedding(context_words).size : (batch_size, window * 2, vocab_size) e.g. [2000, 4, 93]
        # torch.mean(self.embeddings(context_words), dim=1).size : (batch_size, vocab_size) e.g. [2000, 93]

    def forward(self, context_words):
        out = torch.mean(self.embeddings(context_words), dim=1)
        out = self.linear(out)
        return out