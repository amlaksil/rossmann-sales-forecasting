#!/usr/bin/python3
"""
This module contains a class called 'ModuleEvaluator'
that evaluate a machine learning model on a given dataset.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error


class ModelEvaluator:
    """
    A class used to evaluate a machine learning model on a given dataset.

    Attributes
    ----------
    data : DataFrame
        the dataset for evaluation
    model : sklearn model
        the trained machine learning model
    """
    def __init__(self, data_path, model_path):
        """
        Initializes the ModelEvaluator with data and a trained model.
        """
        self.data = pd.read_csv(data_path)
        self.model = joblib.load(model_path)

    def evaluate(self):
        """
        Evaluates the model using the test split of the dataset and
        prints the RMSE.
        """
        X = self.data.drop(['Sales', 'Date'], axis=1)
        y = self.data['Sales']
        _, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'RMSE: {rmse}')
