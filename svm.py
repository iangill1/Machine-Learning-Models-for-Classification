import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#wildfire datasets
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year", "temp", "humidity", "rainfall", "drought_code", "buildup_index", "day", "month", "wind_speed"]
dependent_cols = "fire"

#loading in training set with pandas library
wf_training = pd.read_csv(training_file)
wf_test = pd.read_csv(test_file)
#print(ds_training)
#print(wf_training.shape)

#setting up a matrix for independent variables from training set
xmatrix_training = wf_training.loc[:, independent_cols]
xmatrix_test = wf_test.loc[:, independent_cols]
#print(xmatrix_training.head())

#setting up vector for dependent var from training set
ymatrix_training = wf_training.loc[:, dependent_cols]
ymatrix_test = wf_test.loc[:, dependent_cols]
#print(ymatrix_training.head())

#import SVM package
from sklearn import metrics
from sklearn.svm import SVC
#create a model and use default values for hyperparameters first
svm = SVC(C=1.0, gamma='scale')
svm.fit(xmatrix_training, ymatrix_training)

#calculate prediction for training set and test set
training_prediction = svm.predict(xmatrix_training)
test_prediction = svm.predict(xmatrix_test)

#calculate accuracy of prediction
training_accuracy = metrics.accuracy_score(ymatrix_training, training_prediction)
test_accuracy = metrics.accuracy_score(ymatrix_test, test_prediction)

#print result
print("\ntraining set accuracy with default hyperparameters: ", training_accuracy)
print("test set accuracy with default hyperparameters: ", test_accuracy)