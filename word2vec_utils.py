import os
import csv
from gensim.models.callbacks import CallbackAny2Vec



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