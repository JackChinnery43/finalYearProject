import pandas as pd
import numpy as np
import csv
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime as dt
from dateutil import parser

class marketData:

    def loadMarketData(self):
        # forex trading historical data 
        self.bitcoin_market_data = pd.read_csv('datasets/bitcoinHistoricalData.csv')

        # create a new column 'date' to store the Timestamp column data, but coverted to datetime format
        # then set the new date column to be the index
        self.bitcoin_market_data['date'] = pd.to_datetime(self.bitcoin_market_data.Timestamp, unit='s')
        self.bitcoin_market_data = self.bitcoin_market_data.sort_values(by='date')
        self.bitcoin_market_data = self.bitcoin_market_data.set_index('date')

        # set the datetime range in line with the historical tweets dataframe
        start_date = '2018-05-19 00:00:00'
        end_date = '2019-11-23 23:59:00'
        self.bitcoin_market_data = self.bitcoin_market_data.loc[start_date : end_date]

        # resample data from one minute to one hour
        self.bitcoin_market_data = self.bitcoin_market_data.resample('60T').mean()

        # remove columns with non-related data
        self.bitcoin_market_data.drop(['Timestamp', 'Weighted_Price', 'Volume_(BTC)', 'Volume_(Currency)'], axis=1, inplace=True)

        # rename required columns
        self.bitcoin_market_data.columns = ['Open', 'High', 'Low', 'Close']

        # new column to store the high/low %
        self.bitcoin_market_data['%_Change_High_Low'] = (self.bitcoin_market_data['High'] - self.bitcoin_market_data['Low']) / self.bitcoin_market_data['Close'] * 100
        # new column to store the percentage change between open and close
        self.bitcoin_market_data['%_Change_Open_Close'] = (self.bitcoin_market_data['Close'] - self.bitcoin_market_data['Open']) / self.bitcoin_market_data['Open'] * 100

        self.bitcoin_market_data['Close'].plot()
        plt.title('Daily Close Market Value for Bitcoin (BTC)')
        plt.tight_layout()
        plt.grid()

        self.bitcoin_market_data.to_csv('datasets/finalBitcoinMarketDataHour.csv')
        print(" ")
        print(self.bitcoin_market_data)

        return self.bitcoin_market_data

def main():
    controller = marketData()
    controller.historicalBitcoinMarketData = controller.loadMarketData()

if __name__ == "__main__":
	main()