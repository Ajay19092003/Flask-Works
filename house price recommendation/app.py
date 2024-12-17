import streamlit as st
import joblib
import numpy as np
 
model =joblib.load("model.pkl")
st.title("House Price Prediction App")

st.divider()


st.write("This app uses Machine Learning model for predicting the house price with given features of the house. For using this App you can Enter the inputs from this UI and use the predict buttton")

st.divider()

bedrooms= st.number_input("Number of the bedrooms",min_value=0, value=0)
bathrooms = st.number_input("Number of the bathrooms",min_value=0, value=0)
livingarea = st.number_input("Living area size",min_value=0, value=2000)
Condition =st.number_input("Condititon",min_value=0, value=3)
Schools =st.number_input("Number of the Schools near by",min_value=0, value=0)

st.divider()

x = [[ bedrooms,bathrooms,livingarea,Condition,Schools]]

predictbutton = st.button("Predict")

if predictbutton:

    x_array= np.array(x)
    
    prediction =model.predict(x_array)[0]
    
    st.write(f"Predicted price:{prediction:,.2f}")
        
else:
    st.write("Please use the predict button after entering the values")






 #order of x['number of bedrooms', 'number of bathrooms', 'living area',
       #'condition of the house', 'Number of schools nearby']