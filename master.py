# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:26:09 2018

@author: Jan-Gunther Gosselke
"""

import loading_data as ld
#import NN
import descriptive as de


weather_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/Weather_ALL.csv"
stock_path = "C:/Users/Jan-Gunther Gosselke/Google Drive/SDA/Data/StockIndices.csv"
price_data_list = ld.load_data(stock_path, weather_path)
#return_data_list = log()

##Descriptive Statistics
for data in price_data_list:
    de.temp_descriptive(data)
    de
    
#for data in data_list