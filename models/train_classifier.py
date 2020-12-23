# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
import pickle




import sys


def load_data(database_filepath):
    """
    read data from the database and separate it to features and labels, it also extract categories' names
    :param database_filepath: the path to the database
    :return:
    ndarray: contain the features
    ndarray: contain the labels
    array: contains the categories' names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_responses', engine)
    X = df.iloc[:, :4].values
    Y = df.iloc[:, 4:].values
    categories_names= df.columns[4:]

    return X, Y, categories_names



def tokenize(text):
    """
    tokenize the text into words and lemmatize it
    :param text:
    :return:
    list: list of the cleaned words
    """
    tokens = word_tokenize(text)
    lem = WordNetLemmatizer()

    clean_tokens = [lem.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens


def build_model():
    """
    build the classification model using pipeline and GridSearch
    :return:
    GridSearchCV: the classification model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__max_df': [0.5, 0.75, 1],
        'vect__max_features': [None, 5000, 10000],
        'tfidf__use_idf': [True, False]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    make the prediction using X_test then print the evaluation of the prediction of each category
    :param model: the classification
    :param X_test: ndarray of the features test data
    :param Y_test: ndarray of the labels test data
    :param category_names: list of the categories' names
    """
    Y_pred = model.predict(X_test)

    for cat_idx in range(Y_test[0, :].shape[0]):
        target_names = [f'{category_names[cat_idx]}-{i}' for i in pd.Series(Y_test[:, cat_idx]).unique()]
        print(classification_report(Y_test[:, cat_idx], Y_pred[:, cat_idx], zero_division=0, target_names=target_names))

def save_model(model, model_filepath):
    """
    write the classification model to a pickle file
    :param model: the classification model
    :param model_filepath: the path to the file
    """
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X[:, 1], Y, test_size=0.2)
        
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