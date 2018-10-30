#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# IMPORTS
import quandl
import datetime as dt
import pandas as pd

def getStock(userAPI):
    # Set up Quandl
    quandl.ApiConfig.api_key = userAPI

    # ---- INPUTS -----
    ticker = str(input("Please enter a Ticker:\n> "))
    print("Thanks, please enter the required dates using the following format: 'dd/mm/YYYY'.")
    start = str(input("Now please enter a starting date:\n>"))
    end = str(input("One last step.  Please enter an ending date. For today's day, please type 'today':\n>"))

    # Check the validity of the inputs
    # Checks if the start/end have a / element + if length of each element is ok

    print("Thanks, we are procedding your inputs...\n")

    # Transfrom into date objects

    start = dt.datetime.strptime(start, "%d/%m/%Y")
    if end == "today":
        end = dt.datetime.today()
    else:
        end = dt.datetime.strptime(end, "%d/%m/%Y")

    print("Selected ticker: " + str(ticker))
    print("Starting date is: " + str(start))
    print("Ending date is: " + str(end) + "\n")

    # ---- Get the Data ----

    data = quandl.get_table('WIKI/PRICES', ticker = ticker,
                            qopts = { 'columns': ['ticker', 'date', 'close'] },
                            date = { 'gte': start, 'lte': end },
                            paginate=True)
    # Turn the data into a Pandas DataFrame
    data = pd.DataFrame(data)
    print("Here's what the data look like:\n")
    print(data.head())

    # Set the date as index
    data = data.set_index('date')
    print("\n With date as index")
    print(data.head())

    ## ---- Daily or Weekly ? ----

    # Additionally, user can be asked whether he prefers to have the data in Aggregate
    # weekly or simply in daily.
    # Note that it will override the previous data, to save some memory and time.
    print("\nWould you like to have daily or weekly data ?")
    print(" - Type 1 for DAILY frequency\n - Type 2 for WEEKLY frequency")
    daily = int(input("> "))
    if daily == 2:
        # we now want to Aggregate the data of Volume to weekly
        data['week'] = data.index.week
        data['year'] = data.index.year

        # Group by year and then by week, so that we have a reasonable ordering
        data = data.groupby(['year','week']).mean()
        print("\nWeekly Volumes:")
        print(data.head())

    ## ---- Daily or Weekly ? ---- 
    return data