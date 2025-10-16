import pandas as pd

#wildfire datasets
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year", "temp", "humidity", "rainfall", "drought_code", "buildup_index", "day", "month", "wind_speed"]
dependent_cols = "fire"

#loading in training set with pandas library
df_training = pd.read_csv(training_file)
df_test = pd.read_csv(test_file)
#print(ds_training)
#print(wf_training.shape)

#setting up a matrix for independent variables from training set
xmatrix_training = df_training.loc[:, independent_cols]
xmatrix_test = df_test.loc[:, independent_cols]
#print(xmatrix_training.head())

#setting up vector for dependent var from training set
ymatrix_training = df_training.loc[:, dependent_cols]
ymatrix_test = df_test.loc[:, dependent_cols]
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
c_values = [1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000]
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

#plot results using matlibplot library and numpy library
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(c_values))
width = 0.4
plt.figure(figsize=(10,6))
plt.bar(x - width/2, training_accuracy_c, width, label='Training accuracy')
plt.bar(x + width/2, test_accuracy_c, width, label='Test accuracy')

plt.xticks(x, c_values, rotation=45)
plt.xlabel("C-value")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Effect of different C-values on accuracy of SVM")
plt.legend()
plt.tight_layout()
plt.show()


#manually try different 'gamma' values
gamma_values = [0.0001,0.001,0.01,0.1,1,10,100,1000]
print("\n\n")
print(gamma_values)

training_accuracy_gamma = []
test_accuracy_gamma = []
for g in gamma_values:
    svm_gamma = SVC(C=1, gamma=g)
    svm_gamma.fit(xmatrix_training, ymatrix_training)

    #calculate predictions for training and test dataset
    training_prediction_gamma = svm_gamma.predict(xmatrix_training)
    test_prediction_gamma = svm_gamma.predict(xmatrix_test)

    #calculate accuracy of predictions
    training_accuracy_gamma.append(metrics.accuracy_score(ymatrix_training, training_prediction_gamma))
    test_accuracy_gamma.append(metrics.accuracy_score(ymatrix_test, test_prediction_gamma))

#print results
print(training_accuracy_gamma)
print(test_accuracy_gamma)

plt.figure(figsize=(8,6))
plt.plot(gamma_values, training_accuracy_gamma, marker='o', label='Training accuracy')
plt.plot(gamma_values, test_accuracy_gamma, marker='s', label='Test accuracy')

plt.xscale('log')
plt.xlabel("Gamma value")
plt.ylabel("Accuracy")
plt.title("Effect of different Gamma values on accuracy of SVM")
plt.ylim(0,1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#manually trying different C-values and gamma values
gamma_values1 = [0.0001,0.001,0.01,0.1,1]
c_values1 = [1,10,100,1000,10000]
print("\n\n")
print(gamma_values1)

training_accuracy = []
test_accuracy = []
#for every gamma value, try different C-values
for g in gamma_values1:
    training_acc_for_gamma = []
    test_acc_for_gamma = []

    for c in c_values1:

        svm_gamma = SVC(C=c, gamma=g)

        svm_gamma.fit(xmatrix_training, ymatrix_training)

        #calculate predictions for training and test dataset
        training_prediction_gamma = svm_gamma.predict(xmatrix_training)
        test_prediction_gamma = svm_gamma.predict(xmatrix_test)

        #calculate accuracy of predictions
        training_acc_for_gamma.append(metrics.accuracy_score(ymatrix_training, training_prediction_gamma))
        test_acc_for_gamma.append(metrics.accuracy_score(ymatrix_test, test_prediction_gamma))

    training_accuracy.append(training_acc_for_gamma)
    test_accuracy.append(test_acc_for_gamma)

print(training_accuracy)
print(test_accuracy)

#plot results of accuracy of C-values using different gamma values
plt.figure(figsize=(10,6))

for i,g in enumerate(gamma_values1):
    plt.plot(c_values1, test_accuracy[i], marker='o', label=f'gamma = {g}')

plt.xscale('log')
plt.ylim(0,1.05)
plt.xlabel("C-value")
plt.ylabel("Accuracy")
plt.title("Accuracy with different C-values and Gamma values on the Test Set")
plt.legend(title="Gamma values")
plt.grid(True, which='both', linewidth=0.5)
plt.tight_layout()
plt.show()