import numpy as np
import pandas as pd
import pickle

loaded_model = pickle.load(open('saved_models/customer_classification_model.pkl', 'rb'))
cat_encoder = pickle.load(open('saved_models/categorical_label_encoder.pkl', 'rb'))
num_encoder = pickle.load(open('saved_models/numerical_scaler_encoder.pkl', 'rb'))

# model testing
input_data = (41, 'blue-collar', 'divorced', 'basic.4y', 'unknown', 'yes', 'no',
       'telephone', 'may', 'mon', 1575, 1, 999, 0, 'nonexistent', 
       93.994, -36.4, 4.857
        )

numerical_data = []
cat_data = []
for i in range(len(input_data)):
    if type(input_data[i]) !=int and type(input_data[i]) != float:
        cat_data.append(input_data[i])
    else:
        numerical_data.append(input_data[i])

#categorical data transformation
cat_data_arr = np.asarray(cat_data)
reshaped_cat_data = cat_data_arr.reshape(1,-1)

cat_encoder.transform(reshaped_cat_data)
x_cat_enc_test = cat_encoder.transform(reshaped_cat_data)

# numerical data transformation
num_data_arr = np.asarray(numerical_data)
reshaped_num_data = num_data_arr.reshape(1,-1)

num_encoder.transform(reshaped_num_data)
x_num_enc_test = num_encoder.transform(reshaped_num_data)

#data concatenation
final_input_arr = np.concatenate((x_cat_enc_test, x_num_enc_test), axis= 1)

prediction = loaded_model.predict(final_input_arr)

if prediction[0] == 0:
    print("Result of prediction is 0...")
else:
    print("Result of prediction is 1...")