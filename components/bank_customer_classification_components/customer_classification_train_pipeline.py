import pandas as pd
import numpy as np
import pickle
# Data transformation
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
# Features Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
# Classificators
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("datasets/bank-additional-full.csv", sep=";")
print(len(data))

X = data.iloc[:, :-1]
y = data.iloc[:,-1]

# categorical data preprocessing.......
col_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome']

oe = OrdinalEncoder()
oe.fit(X[col_cat])
X_cat_enc = oe.transform(X[col_cat])
X_cat_enc = pd.DataFrame(X_cat_enc)
X_cat_enc.columns = col_cat

#numerical data preprocessing...
col_num = ['age', 'duration', 'campaign', 'pdays',
       'previous', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m']
        
scaler = MinMaxScaler(feature_range=(0, 1))
X_num_enc = scaler.fit_transform(X[col_num])
X_num_enc = pd.DataFrame(X_num_enc)
X_num_enc.columns = col_num

# data concatenation
x_enc = pd.concat([X_cat_enc, X_num_enc], axis=1)

# encoding the y data
le = LabelEncoder()
le.fit(y)
y_enc = le.transform(y)
y_enc = pd.Series(y_enc)
y_enc.columns = y.name

#train test split
X_train, X_test, y_train, y_test = train_test_split(x_enc, y_enc, test_size=0.33, random_state=1)

# model trainning
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
yhat = model.predict(X_test)
accuracy = accuracy_score(y_test, yhat)
print('Accuracy: %.2f' % (accuracy*100))


# model saving
model_filename = 'customer_classification_model.pkl'
pickle.dump(model, open(model_filename, 'wb'))

# encoders saving
ordinal_encoder_filename = 'categorical_label_encoder.pkl'
pickle.dump(oe, open(ordinal_encoder_filename, 'wb'))

scaler_encoder_filename = 'numerical_scaler_encoder.pkl'
pickle.dump(scaler, open(scaler_encoder_filename, 'wb'))


