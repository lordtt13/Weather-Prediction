# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:02:51 2020

@author: Tanmay Thakur
"""
import pickle
import numpy as np
import tensorflow as tf


X_train, y_train = pickle.load(open( "data.pickle", "rb" ))

model = tf.keras.models.load_model("recurrent_model_initial.h5")

i = np.random.choice(X_train)
p = model.predict(X_train[i].reshape(1, X_train.shape[1], X_train.shape[2]))[0]