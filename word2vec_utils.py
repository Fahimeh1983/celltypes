import os
import csv
from gensim.models.callbacks import CallbackAny2Vec
from numpy import exp, dot,  sum as np_sum


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, filename, filedir):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.filename = filename
        self.filedir = filedir
        self.filepath = os.path.join(filedir, filename)

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        #print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        with open(self.filepath, "a") as myfile:
            writer = csv.writer(myfile, delimiter=',')
            writer.writerow((str(self.epoch),str(loss_now)))
        myfile.close()
        self.epoch += 1

def predict_output_probability(model, W1, context_words_list):
        """Get the probability distribution of the center word given context words.

        Parameters
        ----------
        context_words_list : list of str
            List of context words.
        topn : int, optional
            Return `topn` words and their probabilities.

        Returns
        -------
        list of (str, float)
            `topn` length list of tuples of (word, probability).

        """
        if not model.negative:
            raise RuntimeError(
                "We have currently only implemented predict_output_word for the negative sampling scheme, "
                "so you need to have run word2vec with negative > 0 for this to work."
            )

        if not hasattr(model.wv, 'vectors') or not hasattr(model.trainables, 'syn1neg'):
            raise RuntimeError("Parameters required for predicting the output words not found.")

        word_vocabs = [model.wv.vocab[w] for w in context_words_list if w in model.wv.vocab]
        if not word_vocabs:
            print("All the input context words are out-of-vocabulary for the current model.")
            return None

        word2_indices = [word.index for word in word_vocabs]

        l1 = np_sum(W1[word2_indices], axis=0)
        if word2_indices and model.cbow_mean:
            l1 /= len(word2_indices)

        # propagate hidden -> output and take softmax to get probabilities
        prob_values = exp(dot(l1, model.trainables.syn1neg.T))
        prob_values /= sum(prob_values)
        top_indices = [model.wv.vocab[i].index for i in model.wv.vocab.keys()]

        return [(model.wv.index2word[index1], prob_values[index1]) for index1 in top_indices]