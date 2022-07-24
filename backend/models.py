'''
Author: Morris LaGrand
Date: July, 2022

A python class that creates a series of simple binary classifiers
'''
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class ModelOrchestrator:
    def __init__(self, df, test_size=.1):
        self.df = df
        self.test_size = test_size
        self.features = ['sex', 'age', 'pclass_id', 'sibsp', 'parch']
        self.label = "survived"
        self.models = {"log_reg": LogisticRegression(), "svm": SVC(), "knn": KNeighborsClassifier(n_neighbors=4)}
        # Create training/testing data
        self._split_data()

    def _prepare_df(self):
        self.df = self.df[self.features + [self.label]].dropna()
        self.y = self.df[self.label]
        self.df_features = self.df[self.features]
        self.X = self.df_features.values

    def _split_data(self):
        self._prepare_df()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                                random_state=42)
    def train_model(self, model):
        f = self.models[model]
        clf = f.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, preds, average='weighted')
        return prec, rec, f1
