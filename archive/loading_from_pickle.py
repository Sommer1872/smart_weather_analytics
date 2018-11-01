#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# In[5]:


nations = ["US", "CH", "JPN", "UK"]

tables = []

for nation in nations:
    filename = "./data/" + nation + 'data.pickle' 
    with open(filename, 'rb') as handle:
        df = pickle.load(handle)
        tables.append(df)


# In[6]:


us, ch, jpn, uk = tables 

