import pandas as pd
import numpy as np
import csv
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime as dt
from dateutil import parser

# forex trading historical data 
bitcoin_market_data = pd.read_csv('bitcoinHistoricalData.csv')

# create a new column 'date' to store the Timestamp column data, but coverted to datetime format
# then set the new date column to be the index
bitcoin_market_data['date'] = pd.to_datetime(bitcoin_market_data.Timestamp, unit='s')
bitcoin_market_data = bitcoin_market_data.sort_values(by='date')
bitcoin_market_data = bitcoin_market_data.set_index('date')

# set the datetime range in line with the historical tweets dataframe
start_date = '2018-05-19 00:00:00'
end_date = '2019-11-23 23:59:00'
bitcoin_market_data = bitcoin_market_data.loc[start_date : end_date]

# resample data from one minute to one hour
bitcoin_market_data = bitcoin_market_data.resample('60T').mean()

# remove columns with non-related data
bitcoin_market_data.drop(['Timestamp', 'Weighted_Price'], axis=1, inplace=True)

# rename required columns
bitcoin_market_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']

# new column to store the high/low %
bitcoin_market_data['High/Low_%'] = (bitcoin_market_data['High'] - bitcoin_market_data['Low']) / bitcoin_market_data['Close'] * 100
# new column to store the percentage change between open and close
bitcoin_market_data['%_Change_Open_Close'] = (bitcoin_market_data['Close'] - bitcoin_market_data['Open']) / bitcoin_market_data['Open'] * 100

bitcoin_market_data['Close'].plot()
plt.title('Daily Close Market Value for Bitcoin (BTC)')
plt.tight_layout()
plt.grid()

bitcoin_market_data.to_csv('bitcoinMarketDataHour.csv')
print(bitcoin_market_data)

