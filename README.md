
### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This code needs python 3 or latest in order to be executed and was deloped using the following python libraries installed:

- pandas
- numpy
- nltk
- scikit-learn
- sqlalchemy

## Project Motivation<a name="motivation"></a>
The appen disaster data was used to build a machine learning pipeline to classify disaster response messages into multiple categories. The goal being to assist emergency responders in quickly identifying relevant messages during crisis situations.

The code includes:

- Loading and preprocessing message data from an SQLite database.
- Building a machine learning pipeline using Natural Language Processing (NLP) techniques.
- Training a multi-output classifier based on RandomForestClassifier.
- Hyperparameter tuning using GridSearchCV to optimize model performance.

## File Descriptions <a name="files"></a>
- process_data.py: Python script for loading, merging and cleaning messages and categories datasets then store to SQLite
- train_classifier.py: Python script for building, training and evaluating ML pipeline for the message classifier.
- run.py: Python script for the flask web app that has overview visualization of messages distribution and classifying of mesage by the model as user captures.
- Other supporting files include the SQLite database file containing message data.

## Results <a name="results"></a>
The model achieves good performance in classifying disaster response messages into relevant categories. Evaluation metrics such as precision, recall, and F1-score are used to assess the model's effectiveness.

