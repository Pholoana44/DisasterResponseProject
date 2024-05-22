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

## File Descriptions and commands to execute scripts <a name="files"></a>
- process_data.py: Python script for loading, merging and cleaning messages and categories datasets then store to SQLite. The command below executes the script using disaster message and disaster categories csv in /data directory and then stores the meged cleaned result into Disaster Response SQLite database in the same data directory
  python process_data.py ./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db
  
- train_classifier.py: Python script for building, training and evaluating ML pipeline for the message classifier. it will also generate tuned_model.pkl file which will be used by the web app. Below is a command to execute the script where the database is in /data directory and model is generated and stored in the same directory as the script
  python train_classifier.py ./data/DisasterResponse.db tuned_model.pkl
  
- run.py: Python script for the flask web app that has overview visualization of messages distribution and classifying of message by the model as user captures. This script should be ran after train_classifier.py, when the model file has been generated. Below is the command to execute the script
  python run.py

- The other files include data and templates folder that contains the database, csv input files and htmls for flask web app  
  
## Results <a name="results"></a>
The model achieves good performance in classifying disaster response messages into relevant categories. Evaluation metrics such as precision, recall, and F1-score are used to assess the model's effectiveness.

