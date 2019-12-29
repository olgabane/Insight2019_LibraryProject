import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def plot_actual_v_predicted(predictions_test: np.ndarray, 
	predictions_train: np.ndarray, 
	actual_labels_test: list,
	actual_labels_train: list, 
	figsize: tuple):
	"""
	Prints plots of actual vs predicted values of trained random forest run on test and train sets

	Args: 
		predictions_test: predictions for test set as numpy array
		predictions_train: predictions for train set as numpy array
		actual_labels_test: list of actual values for test set
		actual_labels_train: list of actual values for train set
		figsize: figure size as tuple

	Returns: None
	"""

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)
	ax1.scatter(actual_labels_test, predictions_test, color='black')
	ax2.scatter(actual_labels_train, predictions_train, color='black')
	ax1.set_title("Predictions for test set")
	ax2.set_title("Predictions for train set")
	for ax in [ax1, ax2]:
		ax.set_xlabel("Actual library usage")
		ax.set_ylabel("Predicted library usage")

	# include x=y line
	ax1_max = max(max(actual_labels_test), max(predictions_test))
	ax2_max = max(max(actual_labels_train), max(predictions_train))
	x1=list(range(int(ax1_max+1)))
	x2=list(range(int(ax2_max+1)))
	ax1.plot(x1, x1, ls='-', lw=2, color='red')
	ax2.plot(x2, x2, ls='-', lw=2, color='red')

	plt.show()

	return None

def plot_actual_v_predicted_density(actual_labels: list, predictions: np.ndarray , figsize=(10,8)):   

	"""
	Plot usage vs. predicted usage with density of points overlaid

	Args:
		predictions: predictions for test set as numpy array
		actual_labels: list of actual values for test set
		figsize: figure size as tuple

	Returns: None
	"""

    #Usage vs. Predicted usage with point density overlaid. 
	my_cmap = plt.cm.jet
	my_cmap.set_under('w',0.5)

	plt.rcParams.update({'font.size': 20, 'axes.linewidth': 2, 'ytick.major.width': 2})
	plt.figure(figsize=figsize)
	plt.hist2d(actual_labels, predictions, bins=50, cmap=my_cmap, vmin=0.5)

	# Add colorbar
	cb = plt.colorbar()
	cb.set_label('Libraries', rotation=270, labelpad = 20)

	# Add labels and x=y line
	plt.xlabel("Usage",  labelpad = 8)
	plt.ylabel("Predicted usage",  labelpad = 10)
	# Manually set ticks
	plt.xticks((0, 10, 20, 30, 40, 50), [0, 10, 20, 30, 40, 50])
	#plt.yticks((0, 10, 20, 30, 40, 50), [0, 10, 20, 30, 40, 50])
	x=list(range(55))
	plt.plot(x, x, color = "white")
	# Set y lim if max predicted value is < max actual value
	if max(predictions) < max(actual_labels):
		plt.ylim(0, 55)
	plt.xlim(0, 55)

	plt.show()
    
	return

def plot_feat_importance_lr(lrmodel: LinearRegression, features_list: list, figsize:tuple = (15,4)) -> None:
	"""
	Plot feature importance for trained linear regession

	Args: 
		lrmodel: instance of trained linear regression class
		features_list: features_list in the same order as columns in the dataframe used to train the random forest
	
	Returns: None
	"""

	# Sort by feature importance
	feat_imp_df = pd.DataFrame()
	feat_imp_df['Weight'] = lrmodel.coef_
	feat_imp_df['Feature'] = features_list
	feat_imp_df = feat_imp_df.sort_values(by='Weight', ascending=False)

	# Plot
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	ax.bar(feat_imp_df.Feature, feat_imp_df.Weight, color='black')
	ax.set_xticklabels(feat_imp_df.Feature, rotation=90)
	ax.set_xlabel("Features")
	ax.set_ylabel("Weight")

	plt.show()

	return None

def plot_feat_importance_forest(rfmodel: RandomForestRegressor, features_list: list, figsize:tuple = (15,4)) -> None:
	"""
	Plot feature importance for trained random forest 

	Args: 
		rfmodel: instance of trained random forest regressor class
		features_list: features_list in the same order as columns in the dataframe used to train the ranndom forest
	
	Returns: None
	"""

	# Sort by feature importance
	feat_imp_df = pd.DataFrame()
	feat_imp_df['Importance'] = rfmodel.feature_importances_
	feat_imp_df['Feature'] = features_list
	feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

	# Plot
	fig, ax = plt.subplots(1, 1, figsize=figsize)
	ax.bar(feat_imp_df.Feature, feat_imp_df.Importance, color='black')
	ax.set_xticklabels(feat_imp_df.Feature, rotation=90)
	ax.set_xlabel("Features")
	ax.set_ylabel("Importance")

	plt.show()

	return None

def plot_prediction_within_error(predicted_values:list, 
                                 actual_values:list, 
                                 difference_list:list=[1, 2, 3], 
                                 figsize = (4, 6)) -> None:
    # Difference between actual and predicted 
    errors = [abs(a - b) for a, b in zip(predicted_values,actual_values)]
    
    # Get total samples within error and percent of total
    sum_true = []
    pct_true = []
    for diff in difference_list:
        total_within = sum([error <= diff for error in errors])
        sum_true.append(total_within)
        pct_true.append(total_within/len(errors))
 
    # Plot results
    fix, ax = plt.subplots(1, 1, figsize=figsize)
    #rc.updateParams[''] = 
    ax.bar(difference_list, sum_true, color='black')
    ax.set_yticks([5000, 10000, 12000])
    ax.set_yticklabels(['5K', '10K', '12K'])                     
    ax.set_xlabel("Usage prediction error")
    ax.set_ylabel("Test set libraries")
    
    # Add pct values to each bar
    for i, diff in enumerate(difference_list):
        ax.text(diff, sum_true[i]-1000, str(int(pct_true[i]*100))+'%', color='white', horizontalalignment='center', fontweight='bold')

