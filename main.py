import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#wildfire datasets
training_file = "wildfires_training.csv"
test_file = "wildfires_test.csv"
independent_cols = ["year", "temp", "humidity", "rainfall", "drought_code", "buildup_index", "day", "month", "wind_speed"]
dependent_cols = "fire"

#loading in training set with pandas library
ds_training = pd.read_csv(training_file)
#print(ds_training)
print(ds_training.shape)

#import SVM package
from sklearn.svm import SVC
svc_c = SVC(C=1.0)

