import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import time, json, warnings
from keras.callbacks import Callback

def auc_roc(target, prediction, vocab_size):
    aucs = []
    for c in np.arange(1, vocab_size):
        prediction_c = prediction[:, :, c].ravel()
        target_c = target[:, :, c].ravel()#(target==c).any(axis=-1).ravel()
        if len(set(target_c)) == 2:
            aucs.append(roc_auc_score(target_c, prediction_c))
        else:
            aucs.append(0)
    return np.array(aucs)

class ModelTest(Callback):
    def __init__(self, Xt, Yt, vocab_size, test_every_X_epochs=1, batch_size=500, verbose=1):
        super(Callback, self).__init__()
        self.Xt = Xt #
        self.Yt = np.array(Yt) #
        self.vocab_size = vocab_size #
        self.test_every_X_epochs = test_every_X_epochs #
        self.batch_size = batch_size #
        self.verbose = verbose #

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.test_every_X_epochs != 0:
            return
        prediction = self.model.predict(
            self.Xt, batch_size=self.batch_size, verbose=self.verbose)
        quality = auc_roc(self.Yt, prediction, self.vocab_size)
        print("Mean AUC ROC at epoch %05d: %0.5f" % (epoch, float(quality.mean())))
        print("AUC ROC by classes at epoch %05d:" % (epoch,), quality)