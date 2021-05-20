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

# original tweets dataset from Kaggle
tweets_df = pd.read_csv('datasets/originalTweets.csv', delimiter=';', skiprows=0, lineterminator='\n')

# convert the date/time column to pandas formatting, and set the date column to the index
tweets_df['date'] = pd.to_datetime(tweets_df['timestamp']).dt.tz_convert(None)
tweets_df = tweets_df.sort_values(by='date')
tweets_df = tweets_df.set_index('date')

# set the date range to line up with historical market data
start_date = '2018-05-19 00:00:00'
end_date = '2019-11-23 23:59:00'
tweets_df = tweets_df.loc[start_date : end_date]

# remove columns with non-related data
tweets_df.drop(['user', 'id', 'fullname', 'url', 'timestamp', 'replies', 'likes', 'retweets'], axis=1, inplace=True)
tweets_df.columns = ['tweetContent']

# clean the data
def cleanTweets(tweet):
    # remove mentions
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#', '', tweet)
    # remove retweets
    tweet = re.sub(r'RT[\s]+', '', tweet)
    # remove links
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    return tweet

tweets_df['tweetContent'] = tweets_df['tweetContent'].apply(str) # convert to string so that data can be cleaned
tweets_df['tweetContent'] = tweets_df['tweetContent'].apply(cleanTweets)

# VADER analyser variable to get sentiment scores
analyser = SentimentIntensityAnalyzer()

# get sentiment scores for each row and add to the table
# a new column for the overall compound score, as well as a column each for negative, neutral and positive scores
tweets_df['compound'] = [analyser.polarity_scores(i)['compound'] for i in tweets_df['tweetContent']]
tweets_df['negative'] = [analyser.polarity_scores(i)['neg'] for i in tweets_df['tweetContent']]
tweets_df['neutral'] = [analyser.polarity_scores(i)['neu'] for i in tweets_df['tweetContent']]
tweets_df['positive'] = [analyser.polarity_scores(i)['pos'] for i in tweets_df['tweetContent']]

# Get average sentiment analysis for each day - the data set currently includes multiple rows of data for each day, which will not line up with the market data
df_vader_twitter_sentiment_avg = tweets_df.resample('60T').mean()

#convert data to a csv file
df_vader_twitter_sentiment_avg.to_csv('datasets/finalTwitterSentimentData.csv')

print(tweets_df)
print(df_vader_twitter_sentiment_avg)