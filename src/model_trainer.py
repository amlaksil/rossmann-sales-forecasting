#!/usr/bin/python3
"""
This module contains a class called `ModelTrainer`
that is used to train a machine learning model for
forcasting sales.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from datetime import datetime
import numpy as np


class ModelTrainer:
    """
    A class used to train a machine learning model for forecasting sales.

    Attributes
    ----------
    train_data : DataFrame
        the training dataset
    test_data : DataFrame
        the test dataset
    model : Pipeline
        the machine learning model pipeline
    """
    def __init__(self, train_data_path, test_data_path):
        """
        Initializes the ModelTrainer with training and test data.
        """
        self.train_data = pd.read_csv(
            train_data_path, dtype={'StateHoliday': str})
        self.test_data = pd.read_csv(
            test_data_path, dtype={'StateHoliday': str})
        self.model = None

    def preprocess_data(self, df, is_train=True):
        """
        Preprocesses the input data by handling missing values, encoding
        categorical variables, and scaling numerical features.

        Args
        ----------
        df : DataFrame
            The input dataframe to preprocess.
        is_train : bool, optional
            A flag indicating whether the data is for training or
        testing (default is True).

        Returns
        -------
        DataFrame
            The preprocessed dataframe.
        """
        # Map 'StateHoliday' values
        mapping = {'0': 1, 'a': 0, 'b': 0, 'c': 0}
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.drop(columns=['Date'])

        df['StateHoliday'] = df['StateHoliday'].replace(mapping)

        # Handle missing values in 'Open' column of test data
        if not is_train:
            df['Open'].fillna(1, inplace=True)

        # One-hot encode categorical features
        categorical_features = ['StoreType', 'Assortment', 'PromoInterval']
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        return df

    def train(self):
        """
        Trains the machine learning model using the training data.
        """
        # Preprocess training data
        self.train_data = self.preprocess_data(self.train_data)

        # Separate features and target variable
        X_train = self.train_data.drop(['Sales'], axis=1)
        y_train = self.train_data['Sales']

        # Column transformer to apply scaling
        numeric_features = X_train.select_dtypes(
            include=['int64', 'int32', 'float64', 'UInt32', 'bool']).columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='passthrough'
        )

        # Create a pipeline with preprocessing and model training
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])

        # Train the model
        self.model.fit(X_train, y_train)

    def save_model(self, path):
        """
        Saves the trained model to the specified path.

        Args
        ----------
        path : str
            The directory path where the model should be saved.

        Returns:
        str: The file path of the saved model.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        model_path = f"{path}/model-{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        return model_path
