# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:21:47 2020

@author: Tanmay Thakur
"""

import pickle

from model import get_model
from tensorflow.keras.optimizers import Adam


X_train, y_train = pickle.load(open( "data.pickle", "rb" ))

model = get_model(X_train)

model.compile(loss = 'mse', optimizer = Adam(lr = 1e-3))

model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.25)

model.save("recurrent_model_initial.h5")