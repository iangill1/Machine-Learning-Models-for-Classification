import pandas as pd

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

#print results
print("\ntraining set accuracy with default hyperparameters: ", training_accuracy)
print("test set accuracy with default hyperparameters: ", test_accuracy)

#manually try different 'C' values first
c_values = [400,500,600,700,800,900]
print("\n\n")
print(c_values)

training_accuracy_c = []
test_accuracy_c = []
for c in c_values:
    svm_c = SVC(C=c, gamma='scale')
    svm_c.fit(xmatrix_training, ymatrix_training)

    #calculate predictions for training and test dataset
    training_prediction_c = svm_c.predict(xmatrix_training)
    test_prediction_c = svm_c.predict(xmatrix_test)

    #calculate accuracy of predictions
    training_accuracy_c.append(metrics.accuracy_score(ymatrix_training, training_prediction_c))
    test_accuracy_c.append(metrics.accuracy_score(ymatrix_test, test_prediction_c))

#print results
print(training_accuracy_c)
print(test_accuracy_c)

#plot results using matlibplot library
import matplotlib.pyplot as plt
plt.scatter(c_values, training_accuracy_c, marker="x")
plt.scatter(c_values, test_accuracy_c, marker="o")
#plt.bar(c_values, training_accuracy_c)
plt.xlim([0, max(c_values)+2])
plt.ylim([0.0, 1])
plt.xlabel("C-value")
plt.ylabel("accuracy")
plt.show()