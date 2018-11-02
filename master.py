# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:26:09 2018
"""
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm import matplotlib.pyplot as plt

from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

# local imports
import NN
import descriptive as de
import loading_data as ld

np.set_printoptions(suppress=True)

weather_path = "./data/Weather_ALL.csv"
stock_path = "./data/StockIndices.csv"

price_data_list, weather_per_city = ld.load_data(stock_path, weather_path)

return_data_list = {}
for index in price_data_list:
    return_data_list[index] = pd.DataFrame(columns=list(price_data_list[index]))
    return_data_list[index].rename(columns={'Price Close':'Return'}, inplace=True)
    for city in price_data_list[index].City.unique():
        ordered_returns = price_data_list[index][price_data_list[index]["City"] == city].sort_values(by="Date")
        ordered_returns.rename(columns={'Price Close':'Return'}, inplace=True)
        ordered_returns["Return"] = np.log(ordered_returns['Return']) - np.log(ordered_returns['Return'].shift(periods=1))
        return_data_list[index] = pd.concat([return_data_list[index], ordered_returns]).dropna(axis=0, how = "any")

# save to pickle if needed
#filename = "./price-data.pickle"
#with open(filename, 'wb') as handle:
#    pickle.dump(price_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
#filename = "./retur-data.pickle"
#with open(filename, 'wb') as handle:
#    pickle.dump(return_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

##Descriptive Statistics for data in price_data_list:
    de.temp_descriptive(weather_per_city)
    
for index in return_data_list:
    print(index)
    de.return_histogram(return_data_list[index]['Return'], index)
    

    
#for data in data_list
min_max_scaler = preprocessing.MinMaxScaler()
price_losses_l1 = {}
price_losses_l2 = {}
price_losses_l2_relu = {}
price_losses_lstm = {}

for price_data in price_data_list:
    #Neural Net Data
    print('------\n------\n------\n' + price_data + '\n------')
    data = price_data_list[price_data].pivot_table(index = ["Date", "Price Close"], columns='City',
                          values = ['Mean Temperature Actual',	'Low Temperature Actual',
                                    'High Temperature Actual',	'Precipitation Actual',	'Wind Speed Actual',
                                    'Relative Humidity Actual']).reset_index(level=['Price Close']).dropna(axis=0, how = "any")
    Y = data['Price Close'].to_frame().values
    X = min_max_scaler.fit_transform(data.drop("Price Close", axis = 1).values)
    print('------\n------\n------\n One hidden layer\n------')
    price_losses_l1[price_data] = NN.Fully_Connected_OneL(X, Y)
    print('------\n------\n------\n Two hidden layers\n------')
    price_losses_l2_relu[price_data] = NN.Fully_Connected_TwoL_relu(X, Y)   
    print('------\n------\n------\n Two hidden layers\n------')
    price_losses_l2[price_data] = NN.Fully_Connected_TwoL(X, Y)  

    ## LSTM
    print('------\n------\n------\n One hidden layer\n------')
    lstm_X = X[2:-2]
    lstm_Y = Y[2:-2]
    samples = list()
    samples_results = list()
    length = 130
    # step over the 5,000 in jumps of 200
    for i in range(0,len(lstm_X),length):
    	# grab from i to i + 200
        sample = lstm_X[i:i+length]
        sample_result = lstm_Y[i:i+length]
        samples.append(sample)
        samples_results.append(sample_result)
    X_array = np.array(samples)
    X_array = X_array.reshape((len(samples), length, 42))
    Y_array = np.array(samples_results)
    Y_array = Y_array.reshape((len(samples_results), length, 1))
    price_losses_lstm[price_data] = NN.build_LSTM(X_array, Y_array, length)


return_losses_l1 = {}
return_losses_l2 = {}
return_losses_l2_relu = {}
return_losses_lstm = {}    
for return_data in return_data_list:
    #Neural Net Data
    
    print('------\n------\n------\n' + return_data + '\n------')
    data = return_data_list[return_data].pivot_table(index = ["Date", "Return"], columns='City',
                          values = ['Mean Temperature Actual',	'Low Temperature Actual',
                                    'High Temperature Actual',	'Precipitation Actual',	'Wind Speed Actual',
                                    'Relative Humidity Actual']).reset_index(level=['Return']).dropna(axis=0, how = "any")
    
    Y = data['Return'].to_frame().values
    X = min_max_scaler.fit_transform(data.drop("Return", axis = 1).values)
    print('------\n------\n------\n One hidden layer\n------')
    return_losses_l1[return_data] = NN.Fully_Connected_OneL(X, Y)
    print('------\n------\n------\n Two hidden layers\n------')
    return_losses_l2_relu[return_data] = NN.Fully_Connected_TwoL_relu(X, Y)   
    print('------\n------\n------\n Two hidden layers\n------')
    return_losses_l2[return_data] = NN.Fully_Connected_TwoL(X, Y) 
    
        
    ## LSTM
    print('------\n------\n------\n One hidden layer\n------')
    lstm_X = X[:-2]
    lstm_Y = Y[:-2]
    samples = list()
    samples_results = list()
    length = 130
    # step over the 5,000 in jumps of 200
    for i in range(0,len(lstm_X),length):
    	# grab from i to i + 200
        sample = lstm_X[i:i+length]
        sample_result = lstm_Y[i:i+length]
        samples.append(sample)
        samples_results.append(sample_result)
    X_array = np.array(samples)
    X_array = X_array.reshape((len(samples), length, 42))
    Y_array = np.array(samples_results)
    Y_array = Y_array.reshape((len(samples_results), length, 1))
    return_losses_lstm[return_data] = NN.build_LSTM(X_array, Y_array, length)
    

    ##OLS
    
    for City in data['Mean Temperature Actual']:
        # GENERATE WEATHER BINARY
        data['VeryCold', City] = np.where(data['Mean Temperature Actual', City] \
            <= np.percentile(data['Mean Temperature Actual', City], q=25), 1, 0)
        data['HeavyRain', City] = np.where(data['Precipitation Actual', City] \
            >= np.percentile(data['Precipitation Actual', City], q=75), 1, 0)
        data['ColdRain', City] = data['HeavyRain', City]*data['VeryCold', City]
        data['VeryCold', City].describe()
        data['HeavyRain', City].describe()
        data['ColdRain', City].describe()
        
        # GENERAGE SEASON BINARY
        data['weekday'] = data.index.dayofweek
        data['month'] = data.index.month
        
        data['Monday'] = np.where(data['weekday'] == 0, 1, 0)
        data['Winter'] = np.where((data['month'] == 11) | 
                               (data['month'] == 12) | 
                               (data['month'] == 1), 1, 0)
        
        data['Summer'] = np.where((data['month'] == 6) | 
                               (data['month'] == 7) | 
                               (data['month'] == 8), 1, 0)
        
        data['DayDiff',City] = data['High Temperature Actual', City] - data['Low Temperature Actual', City]
        
        # MODEL1:
        # return_t = return_t-1 + binaryCOLD + binaryRAIN + interactionCOLDRAIN + binaryWINTER + binarySOMMER + binaryMONDAY
        
        data['LagReturn'] = data['Return'].shift(periods=-1)
        data_cut = data[1:]
        X = data_cut[[('LagReturn',''),('VeryCold', City),('HeavyRain', City),('ColdRain', City),('Monday',''),('Winter',''),('Summer','')]]
        Y = data_cut['Return'].to_frame().values
        
        # Regressions:
        reg1 = sm.OLS(endog=Y, exog=X, missing='drop')
        results1 = reg1.fit()
        print(results1.summary())
        
        # MODEL2:
        # return_t = return_t-1 + DayDifference + binaryRAIN + binaryWINTER + binarySOMMER + binaryMONDAY
        
        # Choice of exogenous variables:
        X = data_cut[[('LagReturn',''), ('DayDiff', City), ('HeavyRain', City), ('Monday', ''), ('Winter', ''), ('Summer', '')]]
        
        # Regressions:
        reg2 = sm.OLS(endog=Y, exog=X, missing='drop')
        results2 = reg2.fit()
        print(results2.summary())
        
        # ARMA #####################
        # acf & pacf
        plot_acf(data_cut['Return'],lags=10)
        plot_pacf(data_cut['Return'], lags=10)
        
        # ARMA11 NO exogenous variables
        model = SARIMAX(endog = Y, order=(1,0,1), enforce_stationarity=False)
        model_fit = model.fit(disp=1)
        print(model_fit.summary())
        
        # Plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        plot_acf(residuals,lags=10)
        plt.show()
        residuals.plot(kind='kde')
        plt.show()
        print(residuals.describe())
        
        #ARMA11 with exogenous variables
        X = data_cut[[('DayDiff',City), ('HeavyRain',City), ('Monday',''), ('Winter', ''), ('Summer','')]]
        model2 = SARIMAX(endog = Y, exog = X, order=(1,0,0), enforce_stationarity=False)
        model_fit2 = model2.fit(disp=1)
        print(model_fit2.summary())
        
        # plot residual errors
        residuals2 = pd.DataFrame(model_fit2.resid)
        residuals2.plot()
        plt.show()
        residuals2.plot(kind='kde')
        plt.show()
        print(residuals2.describe())
        
    data['global_cold'] = data["VeryCold"].all(axis = 1)*1
    data['global_rain'] = data["HeavyRain"].all(axis = 1)*1
    data['global_coldrain'] = (data["HeavyRain"].all(axis = 1) & data["VeryCold"].all(axis = 1))*1
    
    # calculate log returns
    Y = data['Return'].to_frame().values
    X = data[[('global_cold', ''), ('global_rain', ''), ('global_coldrain', ''), ('weekday', ''), ('month', ''), ('Monday', ''), ('Winter', '')]]
    model = sm.OLS(Y, X).fit(cov_type='HAC',cov_kwds={'maxlags':20})
    predictions = model.predict(X)
    
    plt.close()
    plt.rc('figure', figsize=(12, 7))
    plt.text(0.01, 0.05, model.summary(), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
