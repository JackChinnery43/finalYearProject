import pandas as pd
import numpy as np
import csv
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime as dt
from dateutil import parser
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

df = pd.read_csv('tweets.csv', delimiter=';', skiprows=0, lineterminator='\n')

df['date'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
df = df.sort_values(by='date')
df = df.set_index('date')

start_date = '2018-05-19 00:00:00'
end_date = '2019-11-23 23:59:00'
df = df.loc[start_date : end_date]

# remove columns with non-related data
df.drop(['user', 'id', 'fullname', 'url', 'timestamp', 'replies', 'likes', 'retweets'], axis=1, inplace=True)
df.columns = ['tweetContent']

# clean the data
def cleanData(text):
    # remove mentions
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    # remove hashtags
    text = re.sub(r'#', '', text)
    # remove retweets
    text = re.sub(r'RT[\s]+', '', text)
    # remove links
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text

df['tweetContent'] = df['tweetContent'].apply(str) # convert to string so that data can be cleaned
df['tweetContent'] = df['tweetContent'].apply(cleanData)

# analyser variable to get sentiment scores
analyser = SentimentIntensityAnalyzer()

# get sentiment scores for each row and add to the table
df['compound'] = [analyser.polarity_scores(i)['compound'] for i in df['tweetContent']]
df['negative'] = [analyser.polarity_scores(i)['neg'] for i in df['tweetContent']]
df['neutral'] = [analyser.polarity_scores(i)['neu'] for i in df['tweetContent']]
df['positive'] = [analyser.polarity_scores(i)['pos'] for i in df['tweetContent']]

# Get average sentiment analysis for each day - the data set currently includes multiple rows of data for each day, which will not line up with the market data
df_vader_twitter_sentiment_avg = df.resample('60T').mean()

#convert data to a csv file
df_vader_twitter_sentiment_avg.to_csv('vaderTwitterSentimentData.csv')

print(df)
print(df_vader_twitter_sentiment_avg)