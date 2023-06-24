"""
movie_db.py -  a python 3 program to manage the movie data model.
"""

__author__ = "Chris Bidlake, Danny Deringer, Lata Gadoo, Jayasri Puppala, Mercy Isaac"
__copyright__ = "Copyright (c) Chris Bidlake, 2021"
__license__ = "GNU GPL3"

# imports
import json

class RatingModel:
    """Class to evaluate reviews using the movie model"""
    def eval(self, input):
        """Default query to evaluate a submitted review"""
        eval = {'input_str': input, 'sentiment': 'Positive'}

        return eval
