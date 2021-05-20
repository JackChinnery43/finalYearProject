Jack Chinnery CI601 Final Year Project

This application looks at the effectiveness of machine learning and natural language processing at predicting future Bitcoin market prices. The application takes two Python files, one for historical Bitcoin finanical market data, and another for historical tweets relating to Bitcoin. The data is cleaned and combined in order to be fed into machine learning algorithms.

File structure:

* datsets folder - the data sets that are used to produce the Bitcoin market price and sentiment scores
* images folder - includes the image of the application high-level view

* Procfile - The terminal command required by Heroku to deploy the application
* setup.sh - file to configure the server
* requirements.txt - the required libraries used within the application

* bitcoinMarketData.py - file to edit and produce the final historical market data csv file
* twitterSentimentVader.py - file to clean and produce the sentiment scores
* combineData.py - file to combine the files produces in the previous two files and create the final data frame used in the machine learning models.
* dataVisualisation.py - the main file that runs the machine learning models, stores the predictions and displays the Streamlit application.

Link to deployed application:
https://jackchinneryfinalyearproject.herokuapp.com/


* Please note within the datasets folder, the original historical Bitcoin market data and historical tweets files are too large to upload to GitHub. This does not impact the final dataVisualisation.py file from working, however if a user wants to run the bitcoinMarketData.py and twitterSentimentVader.py locally, the links to the Kaggle datasets are:

Bitcoin market data:
https://www.kaggle.com/mczielinski/bitcoin-historical-data/metadata

Bitcoin historical tweets:
https://www.kaggle.com/alaix14/bitcoin-tweets-20160101-to-20190329/metadata


The Bitcoin market data will need to be saved from Kaggle and read into line 14 of the bitcoinMarketData.py file.
The historical tweets will need to be saved from Kaggle and read into line 15 of the twitterSentimentVader.py file.
