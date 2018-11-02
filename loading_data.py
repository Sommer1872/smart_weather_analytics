#!/usr/bin/env python
# coding: utf-8

# In[40]:


import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime

# ignore warnings because they are distracting 
import warnings
warnings.filterwarnings('ignore')

# suppress the scientific notation when printing numpy arrays
np.set_printoptions(suppress=True)

dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
# # Loading the financial data

# In[83]:
def load_data(stock_path, weather_path):
    stocks = pd.read_csv(stock_path,
                 sep=';',
                 parse_dates=['Date'],
                 date_parser=dateparse,
                 decimal=',')

    # data cleansing
    stocks['Index'] = [name.replace(".", "") for name in stocks['Index'].values]
    data_per_index = {}
 
    # # Loading the weather data

    weather_per_city = pd.read_csv(weather_path,
                 sep=';',
                 parse_dates=['Date'],
                 date_parser=dateparse,
                 decimal=',')
    cities = pd.DataFrame({'City': ["New York", "Boston", "San Francisco", "Chicago", "London", "Zurich", "Tokyo"],
                              'Country': ["USA", "USA", "USA", "USA", "UK", "Switzerland", "Japan"]})
    weather_per_city = pd.merge(weather_per_city, cities, on="City")
    # look at how many NaNs we have
    # df.isna().sum()
    # drop NaNs
    weather_per_city.dropna(inplace=True)
    
    for stock_index in stocks.Index.unique():
        data_per_index[stock_index] = pd.merge(stocks[stocks["Index"] == stock_index], weather_per_city, on="Date")
    # show all cities
    print([city for city in weather_per_city['City'].unique()])
    # assumes (nrows x ncols) episodes
    return data_per_index, weather_per_city

def main():
    pass

if __name__ == "__main__":
    main()
