# Disaster Response Pipeline Project
This is project 2 for the Udacity Data Science Nano Degree course, where I am looking at analysing disaster data from Apppen to build a model for an API that classifies disaster messages. 
# Project Components

# 1 ETL Pipeline

This process is to load the messages and catergories data set. The datasets are merged, cleaned and stored in the SQLite database.

" In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database" (as stated in the Udacity course)

# 2 ML Pipeline

This process is to write a machine learning pipeline. 

"In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file" (as stated in the Udacity course)

# 3 Flask Web App

This is to create the app.

# File Structure

"- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md" (as stated in the Udacity course)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

4. http://127.0.0.1:3001/

# Credit

Udacity course gave starter code which has been used in this project along with project details used in this readme file. Support has also come from Udacity mentors, Udacity AI, Chatgpt, Stack Overflow



