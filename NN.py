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
k_fold = KFold(n_splits=3)


model = Sequential([
    Dense(32, input_shape=(6,)),
    Dense(1)
])

x = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),
                                             ['Mean Temperature Actual', 'Low Temperature Actual',
                                              'High Temperature Actual', 'Precipitation Actual', 'Wind Speed Actual', 
                                              'Relative Humidity Actual']])
y = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),"Price Close"])
x_test = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),
                                             ['Mean Temperature Actual', 'Low Temperature Actual',
                                              'High Temperature Actual', 'Precipitation Actual', 'Wind Speed Actual', 
                                              'Relative Humidity Actual']]).iloc[1000:]
y_test = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),"Price Close"]).iloc[1000:]

model.compile(optimizer='rmsprop', loss='mse')
model.summary()
model.fit(x, y, validation_split=0.33, epochs=150, batch_size=10)