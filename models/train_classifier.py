import pickle
import sys
# import libraries
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt')       
nltk.download('wordnet')      

def load_data(database_filepath):
    """
    This is loads data from SQLite database
    Defines feature and target variables X and Y

    Args:
    database_filepath (str) - path to SQLlite database file

    Returns: x - messaged 

    """
    # load data from database

    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message'] #message
    Y = df.iloc[:, 4:] #to include all columns starting from related
    return X,Y

def tokenize(text):

    """
    This is loads data from SQLite database
    Defines feature and target variables X and Y

    Args:
    database_filepath (str) - path to SQLlite database file

    Returns: x - messaged 

    """
  #Tokenize the text and initialise the lematizer 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

#cleaning the tokens, converting to lowercase, stripping any leading or trailing spaces
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    

    Returns:
        GridSearchCV: .
    """

    # Instantiate transformers and classifiers
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define hyperparameter grid for GridSearchCV
    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
  
  # Predict using the model
    Y_pred = model.predict(X_test)

# 
    for i, category in enumerate(Y_test):
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()