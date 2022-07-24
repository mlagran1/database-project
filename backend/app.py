'''
Author: Morris LaGrand
Date: July, 2022

A Flask REST server for IU Applied Database Technologies final project.

Routes:
- /train (POST): Trains a machine learning model
- /get/all (GET): Returns all data from database
'''
###############################################################################
# Imports                                                                     #
###############################################################################
# System Imports
from flask import Flask, request
import pandas as pd
# Local Imports
from db import DB
from models import ModelOrchestrator

###############################################################################
# Globals                                                                     #
###############################################################################
# Init Database
DBASE = DB()
DBASE.create_database()

# Init Model Orchestrator
DF_ALL = DBASE.get_all_data()
MO = ModelOrchestrator(df = DF_ALL)

# Create Flask App
app = Flask(__name__)

###############################################################################
# Routes                                                                      #
###############################################################################
@app.route("/train", methods=["POST"])
def train():
    '''
    Train a machine learning model (Logistic Regression, SVM, or KNN)

    Parameters:
        N/A
    Returns:
        A dictionary object that contains precision, recall, and f1-score
    '''
    data = request.get_json()
    # Get model key from post body
    model = data["model"]
    prec, recall, f1 = MO.train_model(model)
    # Create response object
    response = {
        "precision": prec,
        "recall": recall,
        "f1": f1
    }
    return response

@app.route("/get/all", methods=["GET"])
def get_all():
    '''
    Return all Data from the SQL database

    Parameters:
        N/A
    Returns:
        A dictionary object that contains all data from database
    '''
    df = DF_ALL
    return df.to_dict()

###############################################################################
# Main                                                                        #
###############################################################################
def main():
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
