"""
movie_api - a python 3 Flask API to analyze movie review data
"""

__author__ = "Chris Bidlake, Danny Deringer, Lata Gadoo, Jayasri Puppala, Mercy Isaac"
__copyright__ = "Copyright (c) Chris Bidlake, 2023"
__license__ = "GNU GPL3"

# imports
import json
from flask import Flask, request, render_template
from application.movie_model import RatingModel

app = Flask(__name__)

#Load Model
rating_model = RatingModel()

@app.route('/')
def api_homepage():
    """Return an html homepage for browser landing."""
    return render_template('home.html')

@app.route('/api/eval', methods=['GET', 'POST'])
def upload():
    eval = rating_model.eval(json.loads(request.json))
    """POST/PUT upload/update data in the database."""
    
    if eval is None:
        response = app.make_response(('Bad request. Could not parse.', 400))
    else:
        response = app.response_class(
        response=json.dumps(eval),
        mimetype='application/json'
        )
    return response
