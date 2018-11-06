import pandas as pd
from datetime import datetime

date = datetime(2018,10,1)

stocks_path = "./data/StockIndices.csv"

stocks = pd.read_csv(
        stocks_path,
        sep=';',
        decimal=',')

stocks['Date'] = pd.to_datetime(stocks['Date'], format='%d/%m/%Y')

stocks = stocks[stocks['Date'] > date]
print(len(stocks))
print(stocks)

path = "./sample_data/sample_StockIndices.csv"
#stocks.to_csv(path, index=False, sep=';', date_format='%d/%m/%Y', decimal=',')


# Weather
weather_path = "./data/Weather_ALL.csv"
weather = pd.read_csv(weather_path,
                     sep=';',
                     decimal=',')
 
weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y')

weather = weather[weather['Date'] > date]
print(len(weather))
print(weather)

path = "./sample_data/sample_Weather_ALL.csv"
#weather.to_csv(path, index=False, sep=';', date_format='%d/%m/%Y', decimal=',')
