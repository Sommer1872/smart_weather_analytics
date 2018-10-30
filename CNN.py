# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:00:43 2018

@author: Jan-Gunther Gosselke
"""

import tensorflow as tf
import keras as ke
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
import loading_data as ld

weather_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/Weather_ALL.csv"
stock_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/StockIndices.csv"
us_merged, swiss_merged, jpn_merged, UK_merged = ld.load_data(stock_path, weather_path)

model = Sequential([
    Dense(32, input_shape=(6,)),
    Dense(1)
])

x_train = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),
                                             ['Mean Temperature Actual', 'Low Temperature Actual',
                                              'High Temperature Actual', 'Precipitation Actual', 'Wind Speed Actual', 
                                              'Relative Humidity Actual']]).iloc[0:100]
y_train = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),"Price Close"]).iloc[0:100]
x_test = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),
                                             ['Mean Temperature Actual', 'Low Temperature Actual',
                                              'High Temperature Actual', 'Precipitation Actual', 'Wind Speed Actual', 
                                              'Relative Humidity Actual']]).iloc[100:]
y_test = pd.DataFrame(data = us_merged.loc[(us_merged["City"] == "Chicago") & (us_merged["Index"] == "SPX"),"Price Close"]).iloc[100:]

model.compile(optimizer='rmsprop', loss='mse')
model.fit(x_train, y_train, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)