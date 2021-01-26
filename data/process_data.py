import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data"""
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on = 'id')

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)

    # tidy up category col names
    row = categories.iloc[0]
    category_colnames = [cat.split('-', 2)[0] for cat in row]
    categories.columns = category_colnames

    for column in categories :
        # set each value to be the last character of the string, 0 or 1
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # drop original categories column from `df` & concatenate with the new `categories` dataframe
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df[~df.duplicated()]
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """Change from long to wide data - returns a df with message categories as column name
    and binary outcomes in the columns"""
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)

    # tidy up category col names
    row = categories.iloc[0]
    category_colnames = [cat.split('-', 2)[0] for cat in row]
    categories.columns = category_colnames

    for column in categories :
        # set each value to be the last character of the string, 0 or 1
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # drop original categories column from `df` & concatenate with the new `categories` dataframe
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df[~df.duplicated()]

    return df


def save_data(df, database_filename):
    """Saves data to SQL lite db of your naming choice."""
    # Upload to sql database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages_db', engine, index = False, if_exists='replace')
    return


def main():
    "Run from the python terminal - must concluse all 4 sys args, below."
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