import pandas as pd
import numpy as np
import csv
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime as dt
from dateutil import parser
from numpy import cov
from scipy.stats import pearsonr

# Combine Twitter sentiment data and market data data tables #
df_sentiment_data = pd.read_csv("datasets/finalTwitterSentimentData.csv")
df_market_data = pd.read_csv("datasets/finalBitcoinMarketDataHour.csv")

# merge the two dataframes on the date as the index
df_combined_data = pd.merge(left=df_market_data, right=df_sentiment_data, on='date')
df_combined_data['date'] = pd.to_datetime(df_combined_data.date)

df_combined_data.set_index('date', inplace=True)

# convert final dataframe into csv file
df_combined_data.to_csv('datasets/combinedData.csv')
# average sentiment score
sentiment_compound_avg = df_combined_data['compound'].rolling(window=720).mean()
# average stock price close 
close_price_avg = df_combined_data['Close'].rolling(window=720).mean()

print(df_combined_data)

