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


#try different C-values
c_values = [0.00001,0.0001,0.001,0.1,1,10,100,1000,10000,100000,10000000,100000000]
print("\n\n")
print(c_values)

training_accuracy_c = []
test_accuracy_c = []

for c in c_values:
    lr_c = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=1000)
    lr_c.fit(xmatrix_training, ymatrix_training)

    #getting prediction
    training_prediction_c = lr_c.predict(xmatrix_training)
    test_prediction_c = lr_c.predict(xmatrix_test)

    #getting accuracy of prediction and appending it to array
    training_accuracy_c.append(metrics.accuracy_score(ymatrix_training, training_prediction_c))
    test_accuracy_c.append(metrics.accuracy_score(ymatrix_test, test_prediction_c))

print(training_accuracy_c)
print(test_accuracy_c)

#trying different penalty values
penalty = ['l1', 'l2', 'elasticnet', None]

print("\n\n")
print(penalty)

training_accuracy_penalty = []
test_accuracy_penalty = []

#loop through every penalty value
#some solver values are not compatible with certain penalty values, so I have to align values
for p in penalty:
    if p == 'l1':
        solver = 'liblinear'
        lr_penalty = LogisticRegression(penalty=p, solver=solver, max_iter=10000)
    elif p == 'l2':
        solver = 'lbfgs'
        lr_penalty = LogisticRegression(penalty=p, solver=solver, max_iter=10000)
    elif p == 'elasticnet':
        solver = 'saga'
        lr_penalty = LogisticRegression(penalty=p, solver=solver, l1_ratio=0.5, max_iter=10000)
    elif p == None:
        solver = 'lbfgs'
        lr_penalty = LogisticRegression(penalty=p, solver=solver, max_iter=10000)

    #train the model
    lr_penalty.fit(xmatrix_training, ymatrix_training)

    #prediction
    training_prediction_penalty = lr_penalty.predict(xmatrix_training)
    test_prediction_penalty = lr_penalty.predict(xmatrix_test)

    #get accuracy of prediction and append
    training_accuracy_penalty.append(metrics.accuracy_score(ymatrix_training, training_prediction_penalty))
    test_accuracy_penalty.append(metrics.accuracy_score(ymatrix_test, test_prediction_penalty))

print(training_accuracy_penalty)
print(test_accuracy_penalty)

#tuning both hyperparameters for better results
c_values1 = [0.01,1,10,100]
penalty1 = ['l1', 'l2', 'elasticnet', None]

print("\n\n")
print("C-values: ", c_values1)
print("penalty values: ", penalty1)

training_accuracy = []
test_accuracy = []

#for every C-value, try different penalty values
for c in c_values1:
    training_acc_for_c = []
    test_acc_for_c = []

    for p in penalty1:
        if p == 'l1':
            solver = 'liblinear'
            lr = LogisticRegression(C=c, penalty=p, solver=solver, max_iter=100000)
        elif p == 'l2':
            solver = 'lbfgs'
            lr = LogisticRegression(C=c, penalty=p, solver=solver, max_iter=10000)
        elif p == 'elasticnet':
            solver = 'saga'
            lr = LogisticRegression(C=c, penalty=p, solver=solver, l1_ratio=0.5, max_iter=10000)
        elif p == None:
            solver = 'lbfgs'
            lr = LogisticRegression(C=c, penalty=p, solver=solver, max_iter=10000)

        lr.fit(xmatrix_training, ymatrix_training)

        #calculate plrediction
        training_prediction_for_c = lr.predict(xmatrix_training)
        test_prediction_for_c = lr.predict(xmatrix_test)

        #calculate accuracy of prediction and append. Round to 5 decimal places
        training_acc_for_c.append(round(metrics.accuracy_score(ymatrix_training, training_prediction_for_c), 5))
        test_acc_for_c.append(metrics.accuracy_score(ymatrix_test, test_prediction_for_c))

    training_accuracy.append(training_acc_for_c)
    test_accuracy.append(test_acc_for_c)

print(training_accuracy)
print(test_accuracy)

#plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))

for i,g in enumerate(penalty1):
    plt.plot(c_values1, training_accuracy[i], marker='o', label=f'penalty = {g}')

plt.xscale('log')
plt.ylim(0,1.05)
plt.xlabel("C-value")
plt.ylabel("Accuracy")
plt.title("Accuracy with different C-values and Penalty values on the Training Set")
plt.legend(title="Penalty values")
plt.grid(True, which='both', linewidth=0.5)
plt.tight_layout()
plt.show()

