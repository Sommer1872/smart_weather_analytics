﻿# Smart Weather Analytics

Apelt, Gosselke, Sommer & Tuchschmid

## Installation instructions
(these steps have been tested on MacOS. For Linux/Windows, please adjust accordingly)

Use your terminal/command line to  navigate to the preferred location on your drive and execute the following command to download the whole project:
```bash
git clone https://github.com/Sommer1872/smart_weather_analytics.git
```

We highly encurage the use of anaconda as a package manager. If you have anaconda, you can simply `cd smart_weather_analytics` and then run:
```bash
conda env create -f requirements.yml
```
This creates a separate environment named `SWA` and automatically installs all the necessary packages.


## Run instructions

The main file is [master.py](master.py). You can run it to train Neural Networks (fully connected and LSTM variants) that try to predict stock movements from weather data. It also runs different regressions to detect a relationship.


## Data
Unfortunately, this code uses proprietary data that we are not allowed to make widely accessible. If interested, please contact the owner of this repo directly.

Otherwise, you can retrieve the data yourself from your preferred provider. You are going to need 2 files:

`StockIndices.csv` has the format:
|Index;|Date;  |Price Close | 
|----------------------|--|-- |
|                      |  | |

and `Weather_ALL.csv` should look like this:
|Date;|City;|Mean Temperature Actual;  |Low Temperature Actual;  |High Temperature Actual;  |Precipitation Actual;  |Wind Speed Actual;  |Relative Humidity Actual  |
|--|--|--|--|--|--|--|--|
|  |  |  |  |  |  |  |  |


