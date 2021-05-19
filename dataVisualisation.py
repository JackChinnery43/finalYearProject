# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import datetime as dt
from dateutil import parser
import csv
from textblob import TextBlob
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# set the streamlit app to wide view by default
st.set_page_config(
    layout="wide",
    )

class BitcoinPrediction:
    def dataSource(self):
        # original dataset including stock price and sentiment score on an hourly basis - 13288 rows
        self.original_data = pd.read_csv('datasets/combinedData.csv')
        return self.original_data

    def sentimentAnalysis(self):
        self.sentiment_data = pd.read_csv('datasets/combinedData.csv.csv', index_col='date')
        return self.sentiment_data

    def chooseFeatures(self):
        features_available = pd.DataFrame(self.data[['Open', 'High', 'Low', 'Close', 'High/Low_%', '%_Change_Open_Close', 'compound', 'negative', 'neutral', 'positive']])
        self.feature_selection = st.multiselect("Select the features to be included in the machine learning model", features_available.columns)

    def chooseMachineLearningModel(self):
        self.machine_learning_selection = st.selectbox("Select the machine learning model", ('Linear Regression', 'Random Forest Regression', 'SVM Regressor', 'Gradient Boosting Regressor'))

    def loadModelData(self):
        data = pd.read_csv('datasets/combinedData.csv.csv')
        # store close price and sentiment score in a dataframe
        machine_learning_df = self.data[self.feature_selection]
        # split_data 10% equates to 1329 (10% of 13288)
        self.split_data = int(math.ceil(len(machine_learning_df) * 0.1)) # equivalent to 10% of the total dataset - this is what is being predicted
        # new column to store the predicted close price
        machine_learning_df['Predicted_Close_Price'] = data['Close'].shift(-self.split_data)

        # The model is trained and tested on 90% of the total dataset (11959 rows)
        # features (X) data set - what is being used for training (previous close price and sentiment score)
        # target (y) data set - what is being used for testing
        # [:-split_data] = everything except the last x (split_data) rows
        X_features = np.array(machine_learning_df.drop(['Predicted_Close_Price'], 1))
        X_features = preprocessing.scale(X_features)
        X_features_scaled = X_features[:-self.split_data]
        machine_learning_df.dropna(inplace = True)
        y_target = np.array(machine_learning_df['Predicted_Close_Price'])

        # X_features_validation_dataset = 1329 rows (the remaining 10% of the total dataset). This is what is being predicted
        # drop everything except for the last x (1329 - split_data) rows of the feature data set
        self.X_features_validation_dataset = X_features[-self.split_data:]
        self.X_features_validation_dataset_series_close = data['Close'][-self.split_data:]

        # Split the feature dataset into train and test sets
        # X_train = 9567 rows (80% of 11959)
        # X_test = 2392 rows (20% of 11959)
        # y_train = 9567 rows (80% of 11959)
        # y_test = 2392 rows (20% of 11959)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_features_scaled, y_target, test_size = 0.2)

    def modelPredictions(self, predictButton):
        # array to store the models prediction results
        self.models_results = []
        if self.machine_learning_selection == 'Linear Regression':
            self.model_name = LinearRegression()
            self.model = self.model_name.fit(self.X_train, self.y_train)
            # predict the remaining 10% of the total dataset
            self.model_predict = self.model_name.predict(self.X_features_validation_dataset)
            self.model_predictions = self.model_predict
            self.models_results.append(self.model_predictions)
            self.accuracy = self.model.score(self.X_test, self.y_test)

        elif self.machine_learning_selection == 'Random Forest Regression':
            self.model_name = RandomForestRegressor()
            self.model = self.model_name.fit(self.X_train, self.y_train)
            # predict the remaining 10% of the total dataset
            self.model_predict = self.model_name.predict(self.X_features_validation_dataset)
            self.model_predictions = self.model_predict
            self.models_results.append(self.model_predictions)
            self.accuracy = self.model.score(self.X_test, self.y_test)

        elif self.machine_learning_selection == 'SVM Regressor':
            self.model_name = SVR()
            self.model = self.model_name.fit(self.X_train, self.y_train)
            # predict the remaining 10% of the total dataset
            self.model_predict = self.model_name.predict(self.X_features_validation_dataset)
            self.model_predictions = self.model_predict
            self.models_results.append(self.model_predictions)
            self.accuracy = self.model.score(self.X_test, self.y_test)

        elif self.machine_learning_selection == 'Gradient Boosting Regressor':
            self.model_name = GradientBoostingRegressor()
            self.model = self.model_name.fit(self.X_train, self.y_train)
            # predict the remaining 10% of the total dataset
            self.model_predict = self.model_name.predict(self.X_features_validation_dataset)
            self.model_predictions = self.model_predict
            self.models_results.append(self.model_predictions)
            self.accuracy = self.model.score(self.X_test, self.y_test)

        # convert the model results array to a dataframe
        models_predictions_dataframe = pd.DataFrame(self.models_results)
        models_predictions_dataframe = models_predictions_dataframe.transpose()
        models_predictions_dataframe.columns = ['Model Results (Predictions)']
        # set the index of the ML predictions dataframe to match the index of the total dataset (11959, 13288). This allows me to merge the two dataframes on the index
        models_predictions_dataframe.index = range((len(self.data) - self.split_data), len(self.data))
        self.models_predictions_dataframe = models_predictions_dataframe

        # validation data - used to compare actual results against the predicted
        self.data.loc[self.data.tail(self.split_data).index, 'Validation (Actual_Close_Price)'] = self.X_features_validation_dataset_series_close
        # combine both data frames
        self.data_with_predictions = pd.concat([self.data, self.models_predictions_dataframe], axis=1)
        self.data_with_predictions = self.data_with_predictions.set_index('date')

        # find the average weekly predicted and actual values. (168 = 24 * 7) - use to view the general trend
        self.model_prediction_dataframe_avg = self.models_predictions_dataframe.rolling(window=168).mean()
        self.X_features_validation_dataset_series_close_avg = self.X_features_validation_dataset_series_close.rolling(window=168).mean()

        # dataframe storing the model predicted close price and the validation (actual) close price
        self.validation_and_predictions_dataframe = pd.concat([self.models_predictions_dataframe, self.X_features_validation_dataset_series_close], axis=1)
        # dataframe storing the 7 day average model predicted close price and the validation (actual) close price
        self.validation_and_predictions_average_dataframe = pd.concat([self.model_prediction_dataframe_avg, self.X_features_validation_dataset_series_close_avg], axis=1)
        self.validation_and_predictions_average_dataframe.columns = ['Model Results (Predictions)', 'Validation (Actual_Close_Price)']

        # regression evaluation metrics
        self.r2Score = r2_score(self.X_features_validation_dataset_series_close, self.model_predictions)
        self.meanAbsoluteError = mean_absolute_error(self.X_features_validation_dataset_series_close, self.model_predictions)
        self.meanSquaredError = mean_squared_error(self.X_features_validation_dataset_series_close, self.model_predictions)

        return self.models_predictions_dataframe, self.model_predictions, self.models_results, self.accuracy, self.data_with_predictions, self.validation_and_predictions_dataframe, self.validation_and_predictions_average_dataframe, self.r2Score, self.meanAbsoluteError, self.meanSquaredError

def main():
    controller = BitcoinPrediction()
    controller.data = controller.dataSource()
    controller.sentiment = controller.sentimentAnalysis()
    st.title("Machine Learning - Bitcoin Price Prediction")
    pathfile = st.sidebar.selectbox("Project Areas", ["Summary", "Project Data", "Sentiment Analysis", "Machine Learning"])

    if pathfile == "Summary":
        st.write("Project Summary")
        st.write("The effectiveness of machine learning techniques on predicting future financial market prices is increasingly being researched, due to the objective nature and speed of machine learning algorithms at finding patterns and trends in large amounts of data. Because of this, they are frequently being used to influence financial decision making.")
        st.write("Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language. Due to its ability to process human words and their context, it is increasingly being used to judge the overall sentiment of large groups of people.")
        st.write("Investor sentiment is a major influence on market prices. The combination of machine learning and NLP can therefore potentially be used to influence financial decision making.")
        st.write(" ")
        st.write("This project will aim to answer the following questions: ")
        st.write("  -  Is there a relationship between public sentiment relating to Bitcoin, and its close market price on any given day?")
        st.write("  -  Does the semantic value of a tweet relating to a Bictoin improve the accuracy score of machine learning models, when attempting to predict future market prices?")
        st.write("  -  Can machine learning and NLP effectively predict future financial market prices?")
        st.write(" ")
        st.write("Project File Structure: ")
        image_col1, image_col2, image_col3 = st.beta_columns([1, 4, 1])
        with image_col2:
            st.image('images/projectStructure.png', use_column_width=True)

    elif pathfile == "Project Data":
        st.write("Bitcoin Market Data: ")
        st.write("  -  Open, High, Low, Close, Volume(BTC), Currency(BTC).")
        st.write("  -  In one minute intervals between 19/05/2018 - 23/11/2019.")
        st.write("  -  Include a column on the high/low percentage, and a column on the percentage change between open/close.")
        st.write(" ")
        st.write("Historical Twitter Data: ")
        st.write("  -  Datetime, tweet content, are only columns required. All others are removed.")
        st.write("  -  16 million tweets between 19/05/2018 - 23/11/2019.")
        st.write("  -  Data is cleaned, removing; Twitter mentions, hashtags, retweets, links.")
        st.write("  -  VADER sentiment analysis library used to provide a positive, neutral, negative and compound sentiment score.")
        st.write(" ")
        st.write("Final Dataset of 13,288 rows used for machine learning models, resampled into hourly intervals: ")
        hide_dataframe = st.empty()
        if not st.checkbox("Hide dataframe"):
            hide_dataframe.dataframe(controller.data)
        
    elif pathfile == "Sentiment Analysis":
        st.write("Sentiment Analysis")
        sentiment_col1, sentiment_col2 = st.beta_columns(2)
        with sentiment_col1:
            sentiment_score_avg = px.line(controller.sentiment_data['compound'].rolling(window=720).mean())
            sentiment_score_avg.update_layout(width=600, title="VADER Sentiment Score - Monthly Average", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
            st.plotly_chart(sentiment_score_avg)
        with sentiment_col2:
            close_price_avg = px.line(controller.sentiment_data['Close'].rolling(window=720).mean())
            close_price_avg.update_layout(width=600, title="Bitcoin Close Price - Monthly Average", legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
            st.plotly_chart(close_price_avg)
        st.write("Correlational Analysis - key findings: ")
        st.write("  -  In 30th August 2018 the sentiment score begins to decrease, from 0.05, to 0.03 in December 2018. This decrease is matched in Bitcoin's market price, which started at over $6000 in September 2018, and decreased to below $4000 in December 2018.")
        st.write("  -  In January 2019 public sentiment appears to become more positive, with the sentiment score beginning at 0.04 and steadily increasing through to April 2019. Bitcoin's price also begins to slightly increase, from $3000 in January 2019 to around $4000 in April 2019.")
        st.write("  -  In April 2019 public sentiment becomes more positive, increasing continuously until mid-July. This again is matched in Bitcoin's price, which increased from April 2019 to July 2019.")
        st.write("  -  Both the sentiment score and stock market price begin to drop in August 2019, before picking up again November 2019.")

    elif pathfile == "Machine Learning":
        st.write("Machine Learning")
        controller.chooseMachineLearningModel()
        controller.chooseFeatures()
        if len(controller.feature_selection) >= 1:
            predictButton = st.button('Predict')
            controller.loadModelData()
            if predictButton:
                models_predictions_dataframe, model_predictions, models_results, accuracy, data_with_predictions, validation_and_predictions_dataframe, validation_and_predictions_average_dataframe, r2score, meanAbsoluteError, meanSquaredError = controller.modelPredictions(predictButton)
                st.write("Model accuracy on test/train data: ", accuracy * 100, "%")
                st.write("Model r2 Score for future predictions: ", r2score)
                st.write("Model Mean Absolute Error for future predictions: ", meanAbsoluteError)
                st.write("Model Mean Squared Error for future predictions: ", meanSquaredError)
                model_graph = px.line(data_with_predictions[['Close', 'Validation (Actual_Close_Price)', 'Model Results (Predictions)']])
                model_graph.update_layout(width=1000, legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))
                st.plotly_chart(model_graph, use_container_width=True)
                machine_learning_col1, machine_learning_col2 = st.beta_columns((1,2))
                with machine_learning_col1:
                    st.write("Predictions and Actual Close Price")
                    st.dataframe(validation_and_predictions_dataframe)
                with machine_learning_col2:
                    model_graph_avg = px.line(validation_and_predictions_average_dataframe)
                    model_graph_avg.update_layout(title="Bitcoin Machine Learning Prediction - 7 day average")
                    st.plotly_chart(model_graph_avg)
        

if __name__ == "__main__":
	main()