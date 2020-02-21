# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:45:27 2020

@author: Tanmay Thakur
"""
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D 

def get_model(X_train):
    inp = Input(shape=(X_train.shape[1],X_train.shape[2]))
    x = LSTM(512, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, return_sequences = True)(inp)
    x = Dropout(0.25)(x)
    x = LSTM(256, activation = 'tanh', recurrent_activation = 'sigmoid', recurrent_dropout = 0, unroll = False, use_bias = True, return_sequences = True)(inp)
    x = Dropout(0.25)(x)
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([max_pool,avg_pool])
    x = Dense(1024)(x)
    x = Dropout(0.25)(x)
    preds = Dense(X_train.shape[2])(x)
    
    model = Model(inp,preds)
    
    return model