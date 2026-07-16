import streamlit as st 
import joblib 
import numpy as np 

@st.cache_resource
def load_assets():
    model = joblib.load('my_model.pkl')
    scaler = joblib .load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()
    

st.title('Cerebral Stroke Prediction Using Random Forest')
st.write("This model predicts whether a patient has a possibility of getting a cerebral stroke using a random forest model.")

gender = st.selectbox('Gender', options=[0, 1, 2],format_func=lambda x: ['Female', 'Male', 'Other'][x])
age = st.number_input('Age', value=30)
hypertension = st.selectbox('Hypertension', options=[0, 1],format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
heart_disease = st.selectbox('Heart Disease', options=[0, 1],format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
ever_married = st.selectbox('Ever Married', options=[0, 1], format_func=lambda x: {0: 'No', 1: 'Yes'}[x])
work_type = st.selectbox('Work Type', options=[0, 1, 2, 3, 4], format_func=lambda x: { 0: 'Govt_job', 1: 'Never_worked', 2: 'Private', 3: 'Self-employed', 4: 'Children'}[x])
residence = st.selectbox('Residence Type', options=[0, 1],format_func=lambda x: {0: 'Rural', 1: 'Urban'}[x])
glucose = st.number_input('Avg Glucose Level', value=100.0)
bmi = st.number_input('BMI', value=25.0)
smoking = st.selectbox('Smoking Status', options=[0, 1, 2, 3],format_func=lambda x: {0: 'Unknown',1: 'Formerly smoked',2: 'Never smoked',3: 'Smokes'}[x])


if st.button('Predict'):
    input_data = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,residence,glucose,bmi,smoking]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
 
    if prediction[0] == 1:
        st.error(f"The model predicts a possibility of stroke (probability: {probability:.1%})")
    else:
        st.success(f"The model predicts no stroke (probability of stroke: {probability:.1%})")