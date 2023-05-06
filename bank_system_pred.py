import pandas as pd
import numpy as np
import streamlit as st
import pickle
from streamlit_option_menu import option_menu

#loading the saved models
credit_card_fraud_model = pickle.load(open('saved_models/credit_card_fraud_model.pkl','rb'))
customer_classification_model = pickle.load(open('saved_models/customer_classification_model.pkl', 'rb'))
cat_encoder = pickle.load(open('saved_models/categorical_label_encoder.pkl', 'rb'))
num_encoder = pickle.load(open('saved_models/numerical_scaler_encoder.pkl', 'rb'))
#sidebar
with st.sidebar:
    selected = option_menu('Banking System Prediction App',
                           ['Bank Customer Classification Module',
                            'Credit Card Fraud Transaction Module'],
                            icons=['person', 'bank'],
                            default_index=0
                        )
    

# bank customer classification module
if selected == 'Bank Customer Classification Module':
    st.title('Bank Customer Classification Module')
    st.subheader('The Application is related to direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. In order to access if the product (bank term deposit) would be subscribed (yes) or not (no) subscribed.')

    col1, col2, col3 = st.columns(3)

    with col1:
        job = st.multiselect(label='Job', options=['housemaid', 'services', 'admin.', 'blue-collar', 'technician',
       'retired', 'management', 'unemployed', 'self-employed', 'unknown',
       'entrepreneur', 'student'])
    with col2:
        marital = st.multiselect(label='Marital Status', options=['married', 'single', 'divorced', 'unknown'])

    with col3:
        education = st.multiselect(label='Education',options=['basic.4y', 'high.school', 'basic.6y', 'basic.9y',
       'professional.course', 'unknown', 'university.degree',
       'illiterate'])
    with col1:
        default = st.multiselect(label='Default', options=['no', 'unknown', 'yes'])
    
    with col2:
        housing = st.multiselect(label='Housing', options=['no', 'yes', 'unknown'])
    
    with col3:
        loan = st.multiselect(label='Loan', options=['no', 'yes', 'unknown'])

    with col1:
        contact = st.multiselect(label='Contact', options=['telephone', 'cellular'])

    with col2:
        month = st.multiselect(label='Month of Contact', options=['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 'mar', 'apr',
       'sep'])
    with col3:
        day_of_week = st.multiselect(label='Day of week contact', options=['mon', 'tue', 'wed', 'thu', 'fri'])

    with col1:
        poutcome = st.multiselect(label='Outcome', options=['nonexistent', 'failure', 'success'])

    with col2:
        age = st.number_input('Age', min_value=10, max_value=100, step=1)
    with col3:
        duration = st.number_input('Duration')
    with col1:
        campaign = st.number_input('Campaign')
    with col2:
        pdays = st.number_input('pdays')
    with col3:
        previous = st.number_input('Previous')
    with col1:
        cons_price_idx = st.number_input('Price Index')
    with col2:
        cons_conf_idx = st.number_input('Conf Index')
    with col3:
        euribor3m = st.number_input('euribor3m')
    

    # code for Prediction
    customer_classification_result = ''

    
    
    # creating a button for Prediction
    
    if st.button('Customer Classification Result'):
        input_data = (job, marital, education, default, housing, loan,
                    contact, month, day_of_week, poutcome, age, duration, campaign, pdays,
                    previous, cons_price_idx,
                    cons_conf_idx, euribor3m)
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

        customer_prediction = customer_classification_model.predict(final_input_arr)                          
        
        if (customer_prediction[0] == 1):
          customer_classification_result = 'Customer will subscribe to the service.....'
        else:
          customer_classification_result = 'Customer is not going to subscribe to the service.....'
        
    st.success(customer_classification_result)
    
    


    
#fraud detection module
if selected == 'Credit Card Fraud Transaction Module':
    st.title('Credit Card Fraud Transaction Module')
    st.subheader('This Application predicts credit card fraud transactions. It was trained only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are Time and Amount.')

    col1, col2, col3 = st.columns(3)
    

    with col1:
        v1 = st.number_input('V1')
    
    with col2:
        v2 = st.number_input('V2')
    with col3:
        v3 = st.number_input('V3')
    
    with col1:
        v4 = st.number_input('V4')
    
    with col2:
        v5 = st.number_input('V5')
    with col3:
        v6 = st.number_input('V6')
    with col1:
        v7 = st.number_input('V7')
    
    with col2:
        v8 = st.number_input('V8')
    with col3:
        v9 = st.number_input('V9')
    
    with col1:
        v10 = st.number_input('V10')
    
    with col2:
        v11 = st.number_input('V11')
    with col3:
        v12 = st.number_input('V12')
    with col1:
        v13 = st.number_input('V13')
    
    with col2:
        v14 = st.number_input('V14')
    with col3:
        v15 = st.number_input('V15')
    
    with col1:
        v16 = st.number_input('V16')
    
    with col2:
        v17 = st.number_input('V17')
    with col3:
        v18 = st.number_input('V18')
    
    with col1:
        v19 = st.number_input('V19')
    
    with col2:
        v20 = st.number_input('V20')
    with col3:
        v21 = st.number_input('V21')
    
    with col1:
        v22 = st.number_input('V22')
    
    with col2:
        v23 = st.number_input('V23')
    with col3:
        v24 = st.number_input('V24')
    with col1:
        v25 = st.number_input('V25')
    
    with col2:
        v26 = st.number_input('V26')
    with col3:
        v27 = st.number_input('V27')
    
    with col1:
        v28 = st.number_input('V28')
    with col2:
        amount = st.number_input('Transaction Amount')

    # code for Prediction
    credit_card_fraud_result = ''
    
    # creating a button for Prediction
    
    if st.button('Credit Card Transaction Result'):
        credit_card_prediction = credit_card_fraud_model.predict([[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11, 
                                                                   v12,v13,v14,v15,v16,v17,v18,v19,v20,
                                                                   v21,v22,v23,v24,v25,v26,v27,v28,amount]])                          
        
        if (credit_card_prediction[0] == 1):
          credit_card_fraud_result = 'Credit Card Fraud Transaction.....'
        else:
          credit_card_fraud_result = 'Regular Credit Card Transaction...'
        
    st.success(credit_card_fraud_result)
    
    
    