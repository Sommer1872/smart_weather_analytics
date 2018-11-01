# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:26:09 2018

@author: Jan-Gunther Gosselke
"""
import pickle
import pandas as pd
import numpy as np
import math
import loading_data as ld
import NN
import descriptive as de
from sklearn import preprocessing


weather_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/Weather_ALL.csv"
stock_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/StockIndices.csv"
price_data_list, weather_per_city = ld.load_data(stock_path, weather_path)
return_data_list = {}
for index in price_data_list:
    return_data_list[index] = pd.DataFrame(columns=list(price_data_list[index]))
    return_data_list[index].rename(columns={'Price Close':'Return'}, inplace=True)
    for city in price_data_list[index].City.unique():
        ordered_returns = price_data_list[index][price_data_list[index]["City"] == city].sort_values(by="Date")
        ordered_returns.rename(columns={'Price Close':'Return'}, inplace=True)
        ordered_returns["Return"] = np.log(ordered_returns['Return']) - np.log(ordered_returns['Return'].shift(periods=-1))
        return_data_list[index] = pd.concat([return_data_list[index], ordered_returns])

filename = "./price-data.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(price_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
filename = "./retur-data.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(return_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

##Descriptive Statistics
#for data in price_data_list:
#    de.temp_descriptive(weather_per_city)
    

    
#for data in data_list
min_max_scaler = preprocessing.MinMaxScaler()
for price_data in price_data_list:
    #Neural Net Data
    print(price_data)
    data = price_data_list[price_data].pivot_table(index = ["Date", "Price Close"], columns='City',
                          values = ['Mean Temperature Actual',	'Low Temperature Actual',
                                    'High Temperature Actual',	'Precipitation Actual',	'Wind Speed Actual',
                                    'Relative Humidity Actual']).reset_index(level=['Price Close']).dropna(axis=0, how = "any")
    Y = data['Price Close'].to_frame().values
    X = min_max_scaler.fit_transform(data.drop("Price Close", axis = 1).values)
    NN.FullyConected(X, Y)
    
for return_data in return_data_list:
    #Neural Net Data
    print(return_data)
    data = return_data_list[return_data].pivot_table(index = ["Date", "Return"], columns='City',
                          values = ['Mean Temperature Actual',	'Low Temperature Actual',
                                    'High Temperature Actual',	'Precipitation Actual',	'Wind Speed Actual',
                                    'Relative Humidity Actual']).reset_index(level=['Return']).dropna(axis=0, how = "any")
    Y = data['Return'].to_frame().values
    X = min_max_scaler.fit_transform(data.drop("Return", axis = 1).values)
    NN.FullyConected(X, Y)