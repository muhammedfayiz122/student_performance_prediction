import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        #report dictionary for record model scores
        report = {}
        for i in range(len(list(models))):
            #Taking each models from models dictionary
            model = list(models.values())[i]

            #model training
            model.fit(X_train, y_train)

            #predicting on train data
            y_train_pred = model.predict(X_train)

            #predicting on test data
            y_test_pred = model.predict(X_test)

            #Model score on train data
            model_score_train = r2_score(y_train, y_train_pred)

            #Model score on test data
            model_score_test =  r2_score(y_test, y_test_pred)

            #Record model score
            report[list(models.keys())[i]] = model_score_test

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    """
    To load previously saved objects
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


