"This a module of the project where data is read from CSV files and processed"
#importing necessary libraries
import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
   Load messages and categories data from csv files

   Args:
   messages_filepath (str): Filepath for the messages csv file
   categories_filepath (str): Filepath for the categories csv file

   Returns:
   DataFrame: Merged DataFrame containing messages and categories data
   """
   # Load messages and categories data from CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories data on 'id' column
    message_categ = messages.merge(categories, how='outer', on=['id'])

    return message_categ


def clean_data(message_categ):
    """
    Clean the DataFrame by splitting categories, converting to numeric, and dropping duplicates

    Args:
    message_categ: DataFrame to be cleaned

    Returns:
    DataFrame: Cleaned DataFrame
    """
    # Split categories into separate columns
    categories = message_categ['categories'].str.split(';', expand=True)

    # Extract column names from the first row
    row = categories.iloc[0]
    # Remove the last 2 characters to get category names
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # Extract the last character (numeric value) and convert to integer

    def extract_last_character_and_convert_to_numeric(value):
        return int(value[-1])

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(
            extract_last_character_and_convert_to_numeric)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original categories column and concatenate cleaned categories
    message_categ.drop(columns=['categories'], inplace=True)
    message_categ = pd.concat([message_categ, categories], axis=1)
    
    # Drop rows where related is 2
    message_categ = message_categ[message_categ['related'] != 2]
    
    # Remove duplicate rows
    message_categ.drop_duplicates(inplace=True)
    assert len(message_categ[message_categ.duplicated()]) == 0
    return message_categ


def save_data(message_categ, database_filename):
    """
    Save the DataFrame to an SQLite database

    Args:
    message_categ : DataFrame to be saved
    database_filename (str): Filename for the SQLite database
    """
    # Create engine to connect to SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')

    # Save DataFrame to database table
    message_categ.to_sql(
        'message_categories_table',
        engine,
        index=False,
        if_exists='replace')

def main():
    "This is the main function"
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "messages_filepath",
        type=str,
        help="Filepath for messages csv file")
    parser.add_argument(
        "categories_filepath",
        type=str,
        help="Filepath for categories csv file")
    parser.add_argument(
        "database_filepath",
        type=str,
        help="Filename for SQLite database")
    args = parser.parse_args()

    # Extract filepaths from command-line arguments
    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath

    # messages_filepath = input("Enter filepath for messages csv file: ")
    # categories_filepath = input("Enter filepath for categories csv file: ")
    # database_filename = input("Enter filename for SQLite database: ")

    # If command-line arguments are provided, use them; otherwise, use default
    # inputs
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # Print information about data loading
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        # Load and merge messages and categories data into a DataFrame
        message_categories = load_data(messages_filepath, categories_filepath)

        # Clean the dataframe
        print('Cleaning data...')
        message_categories = clean_data(message_categories)

        # Save cleaned data to SQLite database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(message_categories, database_filepath)

        print('Cleaned data saved to database!')

    # If command-line arguments are not provided, show instructions to the user
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
