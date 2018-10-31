# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:26:09 2018

@author: Jan-Gunther Gosselke
"""
import pickle
import math
import loading_data as ld
#import NN
import descriptive as de


weather_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/Weather_ALL.csv"
stock_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/StockIndices.csv"
price_data_list, weather_per_city = ld.load_data(stock_path, weather_path)
return_list = {}
for index in price_data_list:
    return_list[index] = pd.DataFrame(columns=list(price_data_list))
    for city in price_data_list[index].City.unique():
        ordered_prices = price_data_list[index][price_data_list[index]["City"] == city].sort_index()
        return_list[index]["City"] == city] = np.log(ordered_prices['Price Close']) - np.log(ordered_prices['Price Close'].shift(periods=-1))

filename = "./data.pickle"
with open(filename, 'wb') as handle:
    pickle.dump(price_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

##Descriptive Statistics
for data in price_data_list:
    de.temp_descriptive(weather_per_city)
    de
    
#for data in data_list