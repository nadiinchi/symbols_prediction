
# coding: utf-8

# In[1]:

import numpy as np
import theano
import theano.tensor as T
import keras


# In[2]:

from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
import my_callbacks
seed = 0


# In[3]:
import numpy as np


# In[4]:


symbols_to_predict = 3

VOCAB_SIZE = 29
# maxlen = 20  # cut texts after this number of words (among top max_features most common words)
start_char = 1
oov_char = 2
index_from = 3
skip_top = 0
test_split = 0.2

train_size = 10000 #!!!
seq_len = 100
test_size = 1000
weight_decay = 1e-3

p_emb = 0.3
p_dense = 0.3


# In[5]:

with open("../data/train.txt") as fin:
    data = [[int(op) for op in line.split(",")] for line in fin.read().split("\n") if line != ""]


# In[6]:

def generate_random_batch(data, batch_size=1000, seq_len=100, symbols_to_predict=3, last=False, all=False,
                          users_probs=None):
    data = [doc for doc in data if len(doc)>=seq_len+2*symbols_to_predict+1]
    if not all:
        if users_probs is None:
            user_ids = np.random.randint(0, len(data), batch_size)
        else:
            user_ids = np.random.choice(np.arange(len(data)), p=users_probs, size=batch_size)
    else:
        user_ids = np.random.permutation(np.arange(len(data)))[:batch_size]
    X_batch = []
    y_batch_pre = []
    for user in user_ids:
        if not last:
            start_idx = np.random.randint(0, len(data[user])-seq_len-2*symbols_to_predict)
        else:
            start_idx = len(data[user])-seq_len-symbols_to_predict
        X_batch.append(data[user][start_idx:start_idx+seq_len])
        y_batch_pre.append(data[user][start_idx+1:start_idx+seq_len+symbols_to_predict+1])
    X_batch, y_batch_pre = np.array(X_batch), np.array(y_batch_pre)
    y_batch = y_batch_pre[:, :seq_len][:, :, np.newaxis]
    for i in range(1, symbols_to_predict):
        y_batch = np.concatenate((y_batch, y_batch_pre[:, i:i+seq_len][:, :, np.newaxis]), axis=2)
    return X_batch, y_batch


# In[7]:

X_train, by = generate_random_batch(data, batch_size=train_size, seq_len=seq_len, symbols_to_predict=3)
#X = X.reshape(X.shape+(1,))
by_extended = np.zeros((by.shape[0]*by.shape[1], VOCAB_SIZE))
by_extended[np.arange(symbols_to_predict*by.shape[0]*by.shape[1]) // symbols_to_predict, 
            by.ravel()] = 1
y_train = by_extended.reshape(by.shape[0], by.shape[1], VOCAB_SIZE)


# In[8]:

X_test, tby = generate_random_batch(data, batch_size=test_size, seq_len=seq_len,
                                         symbols_to_predict=symbols_to_predict,
                                        last=True, all=True)
tby_extended = np.zeros((tby.shape[0]*tby.shape[1], VOCAB_SIZE))
tby_extended[np.arange(symbols_to_predict*tby.shape[0]*tby.shape[1]) // symbols_to_predict, 
            tby.ravel()] = 1
y_test = tby_extended.reshape(tby.shape[0], tby.shape[1], VOCAB_SIZE)


# In[9]:

def multilabel_CE(predictions, targets):
    """
    universal for multiclass and multilabel: depends on target (number of 1s) and predictions
    """
    epsilon = np.float32(1.0e-6)
    one = np.float32(1.0)
    pred = T.clip(predictions, epsilon, one - epsilon)
    return -T.sum((targets * T.log(pred) + (one - targets) * T.log(one - pred)), axis=1)  #*weights[np.newaxis, :]
    #return T.maximum(0, 1-2*(targets-0.5)*predictions)


# In[11]:

print('Build model...')
model = Sequential()
model.add(Embedding(VOCAB_SIZE + index_from, 128, W_regularizer=None, dropout=0))
model.add(LSTM(128, dropout_W=0, dropout_U=0, return_sequences=True))
model.add(Dropout(p_dense))
model.add(Dense(VOCAB_SIZE, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), activation="sigmoid"))

#optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss=multilabel_CE, optimizer=optimiser)


# In[12]:

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
modeltest_1 = my_callbacks.ModelTest(X_train, y_train, VOCAB_SIZE,
                      test_every_X_epochs=1, verbose=0)
modeltest_2 = my_callbacks.ModelTest(X_test, y_test, VOCAB_SIZE, test_every_X_epochs=1, verbose=0)
callbacks_list = [checkpoint, modeltest_1, modeltest_2]


# In[15]:

model.fit(X_train, y_train, nb_epoch=40, batch_size=128, callbacks=callbacks_list, verbose=True)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



