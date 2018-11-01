# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:00:43 2018

@author: Jan-Gunther Gosselke
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras as ke
from keras.models import Sequential
from keras.layers import Dense, Activation
import loading_data as ld
from sklearn.model_selection import KFold, cross_val_score
from keras.callbacks import History 





def FullyConected(x, y):
    #Hyperparameters
    history = History()
    models = {}
    for i in range(1,10):
        number_of_nodes = i*5
        print("Number of Nodes" + str(number_of_nodes))
        model = Sequential([
            Dense(number_of_nodes, input_shape=(42,)),
            Dense(1)
            ])
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()
        model.fit(x, y, validation_split=0.33, epochs=10, batch_size=128)
    print(history)
    
#def RNN:
    #model = 