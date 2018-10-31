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


# # Loading the financial data

# In[83]:
def load_data(stock_path, weather_path):
    stocks = pd.read_csv(stock_path,
                 sep=';',
                 parse_dates=['Date'],
                 index_col=['Date'],
                 decimal=',')

    # data cleansing
    stocks['Index'] = [name.replace(".", "") for name in stocks['Index'].values]

    us_stocks = stocks[stocks['Index'].isin(['SPX', 'NDX', 'IXIC'])]
    swiss_stocks = stocks[stocks['Index'] == 'SSMI']
    jpn_stocks = stocks[stocks['Index'] == 'N225']
    UK_stocks = stocks[stocks['Index'] == 'FTSE']

    # # Loading the weather data

    df = pd.read_csv(weather_path,
                 sep=';',
                 parse_dates=['Date'],
                 index_col=['Date'],
                 decimal=',')

    # look at how many NaNs we have
    # df.isna().sum()

    # drop NaNs
    df.dropna(inplace=True)

    us = df[df['City'].isin(['Boston', 'Chicago', 'New York', 'San Francisco'])]
    switzerland = df[df['City'] == 'Zurich']
    UK = df[df['City'] == 'London']
    japan = df[df['City'] == 'Tokyo']

    # # Joining the data

    us_merged = pd.merge(us, us_stocks, on='Date')
    ch_merged = pd.merge(switzerland, swiss_stocks, on='Date')
    jp_merged = pd.merge(japan, jpn_stocks, on='Date')
    uk_merged = pd.merge(UK, UK_stocks, on='Date')

    # # Histograms per Month

    # show all cities
    print([city for city in df['City'].unique()])
    # assumes (nrows x ncols) episodes
    return us_merged, ch_merged, jp_merged, uk_merged

