# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 23:06:54 2020

@author: Tanmay Thakur
"""

import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler


data = pd.read_csv('weatherHistory.csv')
input_data = data.iloc[:,3:7]

scaler = StandardScaler()
scaler.fit(input_data[:3*len(input_data)//4]) # 0.75 because train_size is 75% of given data
copy = scaler.transform(input_data)

timestep = 24

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()

	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	agg = pd.concat(cols, axis=1)
	agg.columns = names

	if dropnan:
		agg.dropna(inplace=True)
	return agg


train = series_to_supervised(copy).values

def make_data(train, input_data):
    X_train = []
    y_train = []
    for i in range(timestep, len(input_data)-1):
        X_train.append(train[i-timestep:i, :len(input_data.columns)])
        y_train.append(train[i-timestep, len(input_data.columns):])
    return(np.array(X_train), np.array(y_train))

data_dump = make_data(train, input_data)

pickle_out = open("data.pickle","wb")
pickle.dump(data_dump, pickle_out)
pickle_out.close()

pickle_out = open("scaler.pickle","wb")
pickle.dump(scaler, pickle_out)
pickle_out.close()