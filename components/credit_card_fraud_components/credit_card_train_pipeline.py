# Import the libraries we need to use in this lab
import warnings
warnings.filterwarnings('ignore')

# from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time
import gc, sys

data = pd.read_csv("datasets/creditcard.csv")
print(len(data))

n_replicas = 10

# inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(data.values, n_replicas, axis=0), columns=data.columns)
# Train test split
# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = big_raw_data.drop(columns=['Time', 'Class'], axis=1)

# y: labels vector
y = big_raw_data['Class']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 

# print the shape of the features matrix and the labels vector
print('X.shape=', X_train.shape, 'y.shape=', y_train.shape)

#data scaling
scaler = StandardScaler()
std_x_train = scaler.fit_transform(X_train)
std_x_test = scaler.transform(X_test)

# for data set imbalance
w_train = compute_sample_weight('balanced', y_train)

# Model creation
model = DecisionTreeClassifier(max_depth=4, random_state=35)
model.fit(X_train, y_train, sample_weight=w_train)

#model eval
'''
Model EVAL Training data
Accuracy score
'''

x_train_pred = model.predict(std_x_train)
training_data_acc = accuracy_score(y_train, x_train_pred)

print('Accuracy score of training data: ',training_data_acc)

'''
Model EVAL Test data
Accuracy score
'''

x_test_pred = model.predict(std_x_test)
test_data_acc = accuracy_score(y_test, x_test_pred)

print('Accuracy score of training data: ',test_data_acc)


# model saving
filename = 'credit_card_fraud_model.pkl'
pickle.dump(model, open(filename, 'wb'))