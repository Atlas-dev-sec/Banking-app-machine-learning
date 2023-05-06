import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler

loaded_model = pickle.load(open('saved_models/credit_card_fraud_model.pkl', 'rb'))

input_data = (-1.35980713e+00, -7.27811733e-02,  2.53634674e+00,
  1.37815522e+00, -3.38320770e-01,  4.62387778e-01,  2.39598554e-01,
  9.86979013e-02,  3.63786970e-01,  9.07941720e-02, -5.51599533e-01,
 -6.17800856e-01, -9.91389847e-01, -3.11169354e-01,  1.46817697e+00,
 -4.70400525e-01,  2.07971242e-01,  2.57905802e-02,  4.03992960e-01,
  2.51412098e-01, -1.83067779e-02,  2.77837576e-01, -1.10473910e-01,
  6.69280749e-02,  1.28539358e-01, -1.89114844e-01,  1.33558377e-01,
 -2.10530535e-02,  1.49620000e+02)

input_test_data = (
        -0.512349,
             4.827060,
            -7.973939,
             7.334059,
             0.367704,
            -2.055129,
            -2.935856,
             1.431008,
            -4.544722,
           -5.258096,
            5.716319,
           -5.810407,
            0.723293,
          -12.289133,
            0.378773,
           -2.020734,
           -2.039703,
           0.658183,
            0.832574,
            0.804101,
            0.535620,
           -0.459496,
           -0.009364,
           -1.140436,
           -0.006445,
            0.527970,
            0.558881,
            0.126517,
         0.770000
)

#convert to numpy array
input_data_as_arr =np.asarray(input_data)

# reshape data
input_data_reshaped = input_data_as_arr.reshape(1, -1)

# standarize input data
#scaler = StandardScaler()
#std_data = scaler.transform(input_data_reshaped)

prediction = loaded_model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("No credit card fraud...")
else:
    print("Credit card fraud...")