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

df_sentiment_data = pd.read_csv("vaderTwitterSentimentData.csv")
df_market_data = pd.read_csv("bitcoinMarketDataHour.csv")

df_combined_data = pd.merge(left=df_market_data, right=df_sentiment_data, on='date')
df_combined_data['date'] = pd.to_datetime(df_combined_data.date)

df_combined_data.set_index('date', inplace=True)

df_combined_data.to_csv('combinedData.csv')
# average sentiment score
sentiment_compound_avg = df_combined_data['compound'].rolling(window=720).mean()
# average stock price close 
close_price_avg = df_combined_data['Close'].rolling(window=720).mean()

# plot the average sentiment score
plt.subplot(1, 2, 1)
sentiment_compound_avg.plot()
plt.title('VADER Sentiment Score - Monthly Average')
plt.xlabel('Monthly - Date range between 2018-05-19 and 2019-11-23')
plt.ylabel('Compound Sentiment Score')
plt.legend(['Compound Score'], loc='upper left')
plt.grid()
plt.tight_layout()

# plot the average close price
plt.subplot(1, 2, 2)
close_price_avg.plot()
plt.title('Bitcoin Stock Market Close Price - Monthly Average')
plt.xlabel('Monthly - Date range between 2018-05-19 and 2019-11-23')
plt.ylabel('Bitcoin Close Price')
plt.legend(['Close Price'], loc='upper left')
plt.grid()
plt.tight_layout()
plt.show()

correlation_analysis = pd.merge(left=sentiment_compound_avg, right=close_price_avg, on='date')

start_date = '2018-06-17 23:00:00'
end_date = '2019-11-23 23:59:00'
correlation_analysis = correlation_analysis.loc[start_date : end_date]
print(correlation_analysis)
print(sentiment_compound_avg)
