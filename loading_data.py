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


stocks = pd.read_csv("./data/StockIndices.csv",
                 sep=';',
                 parse_dates=['Date'],
                 index_col=['Date'],
                 decimal=',')


# In[85]:


# data cleansing
stocks['Index'] = [name.replace(".", "") for name in stocks['Index'].values]


# In[87]:


us_stocks = stocks[stocks['Index'].isin(['SPX', 'NDX', 'IXIC'])]
swiss_stocks = stocks[stocks['Index'] == 'SSMI']
jpn_stocks = stocks[stocks['Index'] == 'N225']
UK_stocks = stocks[stocks['Index'] == 'FTSE']


# # Loading the weather data

# In[88]:


df = pd.read_csv("./data/Weather_ALL.csv",
                 sep=';',
                 parse_dates=['Date'],
                 index_col=['Date'],
                 decimal=',')


# In[99]:


# look at how many NaNs we have
# df.isna().sum()


# In[90]:


# drop NaNs
df.dropna(inplace=True)


# In[92]:


us = df[df['City'].isin(['Boston', 'Chicago', 'New York', 'San Francisco'])]
switzerland = df[df['City'] == 'Zurich']
UK = df[df['City'] == 'London']
japan = df[df['City'] == 'Tokyo']


# # Joining the data

# In[93]:


us_merged = pd.merge(us, us_stocks, on='Date')
swiss_merged = pd.merge(switzerland, swiss_stocks, on='Date')
jpn_merged = pd.merge(japan, jpn_stocks, on='Date')
UK_merged = pd.merge(UK, UK_stocks, on='Date')


# # Histograms per Month

# In[6]:


# show all cities
print([city for city in df['City'].unique()])


# In[7]:


# assumes (nrows x ncols) episodes
fig, axes = plt.subplots(nrows=4, ncols=3,
                         sharex=True, sharey=True,
                         figsize=(20,20)
                        )

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

for month_i, ax in enumerate(axes.flatten()):
    
    subset = df[df.index.month == month_i+1]

    sns.distplot(subset['Mean Temperature Actual'], kde=True, ax=ax)
    ax.set_title(months[month_i])
    
# Save the full figure...
fig.savefig('./plots/monthly_temperatures.png')

