import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    This function evaluates the given models and returns a dictionary with model names as keys and their R2 scores as values.
    """
    try:
        
        logging.info("Evaluating models...")

        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.error("Error in model evaluation")
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)