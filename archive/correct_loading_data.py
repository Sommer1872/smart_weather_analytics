
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

# ignore warnings because they are not relevant
import warnings
warnings.filterwarnings('ignore')

# suppress the scientific notation when printing numpy arrays
np.set_printoptions(suppress=True)


# # Loading the data

# Stocks
stocks = pd.read_csv("./data/StockIndices.csv",
                 decimal=',')

# stocks date format: 29/10/2018
stocks['Date'] = pd.to_datetime(stocks['Date'], format='%d/%m/%Y')
stocks.set_index('Date', inplace=True)
stocks.drop(columns='Unnamed: 0', inplace=True)
stocks.head()

# converting prices to floats
stocks['Price Close'] = [float(price) for price in stocks['Price Close']]

# data cleansing
stocks['Index'] = [name.replace(".", "") for name in stocks['Index'].values]

# selecting a specific index
stock_index = 'SPX'
stock_index = stocks[stocks['Index'] == stock_index]
stock_index.head()


# Weather
weather = pd.read_csv("./data/Weather_ALL.csv",
                 sep=';',
                 decimal=',')

# weather date format: 29/10/2018
weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y')
weather.set_index('Date', inplace=True)
weather.head()

# drop NaNs
weather.dropna(inplace=True)

weather.head()


# # Merging prices and weather
data = pd.merge(stock_index, weather, on='Date')

