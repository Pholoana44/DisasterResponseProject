import sys
import pandas as pd
from sqlalchemy import create_engine
import argparse

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
     df = messages.merge(categories, how='outer', on=['id'])

     return df

def clean_data(df):
    """
    Clean the DataFrame by splitting categories, converting to numeric, and dropping duplicates
    
    Args:
    df (DataFrame): DataFrame to be cleaned
    
    Returns:
    DataFrame: Cleaned DataFrame
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    #Extract column names from the first row
    row = categories.iloc[0]
    # Remove the last 2 characters to get category names
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # Extract the last character (numeric value) and convert to integer
    def extract_last_character_and_convert_to_numeric(value):
        return int(value[-1])
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(extract_last_character_and_convert_to_numeric)
        
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
    
    #Drop the original categories column and concatenate cleaned categories  
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    #Remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save the DataFrame to an SQLite database
    
    Args:
    df (DataFrame): DataFrame to be saved
    database_filename (str): Filename for the SQLite database
    """
    # Create engine to connect to SQLite database
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save DataFrame to database table
    df.to_sql('message_categories_table', engine, index=False, if_exists='replace')  

def main():
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("messages_filepath", type=str, help="Filepath for messages csv file")
    parser.add_argument("categories_filepath", type=str, help="Filepath for categories csv file")
    parser.add_argument("database_filepath", type=str, help="Filename for SQLite database")
    args = parser.parse_args()
    
    #Extract filepaths from command-line arguments
    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath
    
    #messages_filepath = input("Enter filepath for messages csv file: ")
    #categories_filepath = input("Enter filepath for categories csv file: ")
    #database_filename = input("Enter filename for SQLite database (e.g., 'DisasterResponse.db'): ")
    
    # If command-line arguments are provided, use them; otherwise, use default inputs
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # Print information about data loading
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
           .format(messages_filepath, categories_filepath))
        
        # Load and merge messages and categories data into a DataFrame
        df = load_data(messages_filepath, categories_filepath)
        
        #Clean the dataframe
        print('Cleaning data...')
        df = clean_data(df)
        
        #Save cleaned data to SQLite database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
        
    #If command-line arguments are not provided, show instructions to the user
    else: 
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()