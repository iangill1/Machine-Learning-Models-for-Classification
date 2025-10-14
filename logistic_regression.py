import pandas as pd

#wildfire datasets
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year", "temp", "humidity", "rainfall", "drought_code", "buildup_index", "day", "month", "wind_speed"]
dependent_cols = "fire"

#loading in training set with pandas library
wf_training = pd.read_csv(training_file)
wf_test = pd.read_csv(test_file)


#setting up a matrix for independent variables from training set
xmatrix_training = wf_training.loc[:, independent_cols]
xmatrix_test = wf_test.loc[:, independent_cols]

#setting up vector for dependent var from training set
ymatrix_training = wf_training.loc[:, dependent_cols]
ymatrix_test = wf_test.loc[:, dependent_cols]

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#default hyperparameters
lr = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000)
lr.fit(xmatrix_training, ymatrix_training)

#calculate prediction of training and test sets
training_prediction = lr.predict(xmatrix_training)
test_prediction = lr.predict(xmatrix_test)

#calculate accuracy of predictions
training_prediction_accuracy = metrics.accuracy_score(ymatrix_training, training_prediction)
test_prediction_accuracy = metrics.accuracy_score(ymatrix_test, test_prediction)

print("\ntraining set accuracy with default parameters: ", training_prediction_accuracy)
print("test set accuracy with default parameters: ", test_prediction_accuracy)

