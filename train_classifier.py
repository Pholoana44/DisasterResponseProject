"This is the module that create and train the model"
#Import important libraries
import sys
import re
import argparse
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

# Download NLTK resources
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    """
    Load data from SQLite database and return feature (X), target (Y), and category names.

    Args:
    database_filepath (str): Filepath of SQLite database.

    Returns:
    X (Series): Series containing message data.
    Y (DataFrame): DataFrame containing target data (category labels).
    category_names (list): List of category names.
    """
    # Create engine to connect to SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # Read data from database table into message_categories
    message_categories = pd.read_sql_table('message_categories_table', con=engine)

    # Extract feature (X), target (Y) data and category names
    X = message_categories['message']
    Y = message_categories.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and preprocess text data.

    Args:
    text (str): Input text string.

    Returns:
    tokens (list): List of preprocessed tokens.
    """
    # convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize into words
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [
        token for token in tokens if token not in stopwords.words("english")]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def build_model():
    """
    Build and return a machine learning pipeline with GridSearchCV/RandomizedSearchCV.

    Returns:
    pipeline (Pipeline): Scikit-learn pipeline object with GridSearchCV.
    """
    # Define the pipeline with CountVectorizer, TfidfTransformer, and
    # RandomForestClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Define parameters for grid search
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
        'tfidf__use_idf': [True, False],
        # number of trees in the forest
        'clf__estimator__n_estimators': [50, 100],
        # minimum number of samples required
        'clf__estimator__min_samples_split': [2, 5]
    }
    # Create CridSearchCV object
    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=5,
        verbose=2,
        n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate a multi-label classification model using a pipeline.

    Args:
    model (Pipeline): Trained Scikit-learn pipeline object.
    X_test (Series): Series containing test input data (messages).
    Y_test (DataFrame): DataFrame containing test target data (category labels).
    """
    # Make predictions on test data
    Y_pred = model.predict(X_test)

    # Display classification report for each category
    for i, column in enumerate(Y_test.columns):
        print(f"Report for category: {column}\n")
        print(classification_report(Y_test[column], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save a best model to a pickle file.

    Args:
    model (object): best model object.
    model_filepath (str): Filepath to save the model as a pickle file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    "This is the main function"
    # Parse command-line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "database_filepath",
        type=str,
        help="Filename for SQLite database")
    parser.add_argument(
        "model_filepath",
        type=str,
        help="Filename for model pkl file")
    args = parser.parse_args()

    # Extract database and model filepaths from command-line arguments
    database_filepath = args.database_filepath
    model_filepath = args.model_filepath

    # model
    if len(sys.argv) == 3:
        # Load data from SQLite database
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # Split data into training and testing
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.8, random_state=42)

        # Build the machine learning pipeline
        print('Building model...')
        model = build_model()

        # Train the model
        print('Training model...')
        model.fit(X_train, Y_train)

        # Get the best estimator found by grid search
        # best_model = grid_search.best_estimator_

        # Evaluate the best model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        # save the best model to pickle file
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
