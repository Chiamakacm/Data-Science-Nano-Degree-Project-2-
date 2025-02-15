import sys
print(sys.executable)
import pandas as pd
import  sqlalchemy  
from sqlalchemy import create_engine  #import libaries

def load_data(messages_filepath, categories_filepath):
    """
    This is to load the two CSV files 
    then clean the join to ensure no nan values
    then merge files and join on  id

    Args:
    messages_filepath (str) This is the file path for messages (csv)
    categories_filepath (str) This is the file path for categories(csv)

    Returns: df with the merged data

    """
    messages = pd.read_csv(messages_filepath,)
    categories = pd.read_csv(categories_filepath)
    messages = messages.dropna(subset=['id']) 
    categories = categories.dropna(subset=['id'])  
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner')
    return df

def clean_data(df):
    """
    To clean the data
    Split `categories` into separate category columns
    Convert category values to just numbers 0 or 1
    Replace `categories` column in `df` with new category columns
    Remove duplicates
    

    Args: the df containing messages and categories
 

    Returns: df - cleaned data

    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.astype(str).apply(lambda x: x[:-2] if len(x)> 2 else x)
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1] 
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column].str[-1])

    # drop the original categories column from `df`
    df= df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df




def save_data(df, database_filepath):
   """
   
    Save the clean dataset into an sqlite database

    Args: 
    df - the cleaned data

    Returns: saves the df to into the sqlite database

    """
   engine = create_engine(f'sqlite:///{database_filepath}')
   df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()