#!/usr/bin/python3
"""
This module contains a class called 'Predictor' which is
used to make predictions using a trained machine learning model
"""
import joblib
import pandas as pd


class Predictor:
    """
    A class used to make predictions using a trained machine learning model.

    Attributes
    ----------
    model : sklearn model
        the trained machine learning model
    """
    def __init__(self, model_path):
        """
        Initializes the Predictor with a trained model.
        """
        self.model = joblib.load(model_path)

    def predict(self, data):
        """
        Makes a prediction based on the input data.

        Parameters
        ----------
        data : dict
            A dictionary containing the feature values for the prediction.

        Returns
        -------
        float
            The predicted value.
        """
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)
        return prediction[0]
