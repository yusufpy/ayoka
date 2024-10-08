import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier, Pool

# Load the saved model
@st.cache_resource
def load_model():
    with open('catboost_agg_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app
st.title('Student Performance Prediction')

# Input fields
st.header('Enter Student Information')

# Numeric inputs
total_attendance = st.number_input('Total Attendance', min_value=0, value=180)
total_resource_duration = st.number_input('Total Resource Duration (minutes)', min_value=0, value=500)
number_of_siblings = st.number_input('Number of Siblings', min_value=0, value=2)
years_experience = st.number_input('Years of Experience (for teachers)', min_value=0, value=5)
avg_score = st.number_input('Average Score', min_value=0.0, max_value=100.0, value=70.0)

# Categorical inputs
health_information = st.selectbox('Health Information', ['Excellent', 'Good', 'Fair', 'Poor'])
socioeconomic_status = st.selectbox('Socioeconomic Status', ['Upper','Lower','Middle'])
parent_education_level = st.selectbox('Parent Education Level', ['Tertiary', 'Secondary','Primary'])
family_structure = st.selectbox('Family Structure', ['Both Parents','Single Parent'])
transport_mode = st.selectbox('Transport Mode', ['Walk','Bicycle','Bus'])
qualification = st.selectbox('Qualification (for teachers)', ['M.ED','B.Ed','M.sc'])
boarding_status = st.selectbox('Boarding Status', ['Day Student', 'Boarder'])

# Prediction button
if st.button('Predict Performance'):
    # Prepare input data
    input_data = pd.DataFrame({
        'total_attendance': [total_attendance],
        'total_resource_duration': [total_resource_duration],
        'health_information': [health_information],
        'socioeconomic_status': [socioeconomic_status],
        'parent_education_level': [parent_education_level],
        'number_of_siblings': [number_of_siblings],
        'family_structure': [family_structure],
        'transport_mode': [transport_mode],
        'qualification': [qualification],
        'years_experience': [years_experience],
        'boarding_status': [boarding_status],
        'avg_score': [avg_score]
    })

    # Get categorical feature indices
    cat_features = [input_data.columns.get_loc(col) for col in ['health_information', 'socioeconomic_status', 'parent_education_level', 'family_structure', 'transport_mode', 'qualification', 'boarding_status']]

    # Create a Pool object
    input_pool = Pool(input_data, cat_features=cat_features)

    try:
        # Make prediction
        prediction = model.predict(input_pool)
        prediction_proba = model.predict_proba(input_pool)

        # Display prediction
        st.subheader('Prediction Results')
        st.write(f'Predicted Performance Category: {prediction[0]}')
        
        st.subheader('Prediction Probabilities')
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(proba_df)
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check if the input data matches the model's expectations.")
        st.write("Input Data:")
        st.write(input_data)

# Instructions for running the app
st.sidebar.header('How to use')
st.sidebar.write("""
1. Enter the student's information in the input fields.
2. Click the 'Predict Performance' button.
3. View the predicted performance category and probabilities.
""")

# Add information about the model
st.sidebar.header('About the Model')
st.sidebar.write("""
This app uses a CatBoost model trained on student data to predict performance categories.
The model considers various factors including attendance, resource usage, health information,
socioeconomic status, family structure, and academic scores to make predictions.
""")