"""
movie_db.py -  a python 3 program to manage the movie data model.
"""

__author__ = "Chris Bidlake, Danny Deringer, Lata Gadoo, Jayasri Puppala, Mercy Isaac"
__copyright__ = "Copyright (c) Chris Bidlake, 2021"
__license__ = "GNU GPL3"

# imports
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class RatingModel():
    """Class to evaluate reviews using the movie model"""
    def __init__(self):
        # Load model and vectorizer
        self.nb_model = pickle.load(open('.\model\nb_model.pkl','rb'))
        self.nb_vectorizer = pickle.load(open('.\model\nb_vectorizer.pkl','rb'))

    def eval(self, user_input):
        """Default query to evaluate a submitted review"""
        # Preprocess the review text
        vectorized_review = self.nb_vectorizer.transform([user_input])

        # Make sentiment prediction
        sentiment_prediction = self.nb_model.predict_proba(vectorized_review)
        
        # Print the predicted sentiment
        if sentiment_prediction[0][0] > 0.6:
            sentiment = 'positive'
        else:
            sentiment = 'negative'
        
        eval = {'input_str': user_input, 'sentiment': sentiment, 'predict_proba': sentiment_prediction[0]}

        return eval
