"""
Functions to predict library usage using several models. 
"""

import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def scale_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """
    Fit scaler to scale features to range between 0 and 1 using train set only.
    Scale train and test sets to range between 0 and 1.

    Args:
        df_train: train set features dataframe (without labels)
		df_test: test set features dataframe (without labels)
    Returns:
    	feats_train_scaled: scaled features for training as numpy array 
        feats_test_scaled: scaled features for testing as numpy array 
    """

    # instantiate scaler class
    scaler = MinMaxScaler()

    # Fit scaler using tran set only
    scaler.fit(df_train)

    # Scale features for train and test sets
    feats_train_scaled = scaler.transform(df_train)
    feats_test_scaled = scaler.transform(df_test)

    return feats_train_scaled, feats_test_scaled

def train_test_split(df: pd.DataFrame,
			 unique_id_col_name:str='FSCSKEY',
			 seed_val:int=11,
			 fract_test: float=0.2) -> (pd.DataFrame, pd.DataFrame, list, list):

	"""
	# Get random set of libraries to use as test set.
	# Samples contain data from 7 years. 
	# Therefore, the same library can appear up to7 times. 
	# To ensure that no library is split between train and test set, 
	# 	 (would leak info unfairly), choose random 20% of library IDs for test set


	Args:
	df: dataframe of features and labels
	unique_id_col_name: column name in df that contain unique library IDs
	seed_val: seed for random number 
	fract_test: fraction of unique IDs to use for test set

	Returns:
	X_train: dataframe of training samples
	X_test: dataframe of test samples
	y_train: labels for training samples
	y_test: labels for test samples

	"""
	# Get unique IDs
	unique_ids = set(list(df[unique_id_col_name]))

	# Seed for reproducibility
	random.seed(seed_val)
	# Choose ids for test set
	test_set_ids = random.sample(unique_ids, int(fract_test * len(unique_ids)))

	# Split to train/test
	Train_samples=df[~df[unique_id_col_name].isin(test_set_ids)]
	Test_samples=df[df[unique_id_col_name].isin(test_set_ids)]

	X_train = Train_samples.drop(columns=['usage', unique_id_col_name])
	X_test = Test_samples.drop(columns=['usage', unique_id_col_name])
	y_train = list(Train_samples.usage)
	y_test = list(Test_samples.usage)

	return (X_train, X_test, y_train, y_test)

def train_lr(X_train, y_train) -> LinearRegression:
	"""
	Fit logistic regession class

	Args:
		X_train: dataframe of training samples
		y_train: labels for training samples

	Returns:
		lrmodel: instance of fit LinearRegression class
	"""

	lrmodel = LinearRegression()
	lrmodel.fit(X_train, y_train)

	return lrmodel	

def train_forest(X_train, y_train, n_estimators:int=100, seed_val:int=11, max_depth:int=60) -> RandomForestRegressor :
	"""
	Train random forest regressor

	Args:
		X_train: dataframe of training samples
		y_train: labels for training samples
		n_estimators: number of trees as int
		seed_val: seed for random state
		max_depth: maximum depth of tree as int

	Returns:
		rfmodel: instance of trained RandomForestRegressor class
	"""
	rfmodel = RandomForestRegressor(n_estimators=n_estimators, random_state=seed_val, max_depth=max_depth)
	rfmodel.fit(X_train, y_train)

	return rfmodel

def predict_evaluate_model(model, samples_df: pd.DataFrame, actual_labels: list) -> (list, float):
	"""
	Given trained random forest or linear regression model, make predictions for provided sample.
	Evaluate prediction.

	Args: 
		rfmodel: instance of trained random forest regressor class
		samples_df: dataframe of test samples (ususally X_test)
		actual_labels: list of actual labels values

	Returns: 
		predictions: predictions as list
		score_r2: R^2 score (comparing actual to predicted values)

	"""

	predictions = list(model.predict(samples_df))
	score_r2 = model.score(samples_df, actual_labels)

	return predictions, score_r2


