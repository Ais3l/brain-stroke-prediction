import streamlit as st
import pickle
import sqlite3
import os
import numpy as np
import gdown

# Function to download model from Google Drive
def download_model_from_google_drive(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)

# Google Drive file ID for your lgbmodel.pkl
file_id = '1sIwINxqeO6PBA5WY55_cGwS6OKsXyLvI'
model_output_path = './model/lgbmodel.pkl'  # Path where the model will be downloaded

# Check if the model file exists locally, otherwise download it
if not os.path.exists(model_output_path):
    download_model_from_google_drive(file_id, model_output_path)

# Load the trained LightGBM model
with open(model_output_path, 'rb') as file:
    model = pickle.load(file)

# Define the database path
db_path = './database.db'

def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY, gender INTEGER, age REAL, hypertension INTEGER, heart_disease INTEGER,
                 work_type INTEGER, Residence_type INTEGER, avg_glucose_level REAL, bmi REAL, smoking_status INTEGER,
                 prediction REAL, result TEXT)''')
    conn.commit()
    conn.close()

def alter_table():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute('ALTER TABLE predictions ADD COLUMN result TEXT')
    except sqlite3.OperationalError as e:
        print(f"OperationalError: {e}")
    conn.commit()
    conn.close()

# Initialize the database and alter table if necessary
init_db()
alter_table()

# CSS styles defined as a string
css_styles = """
<style>
body {
    width: 600px;
}

.button {
    padding-top: 20px;
}
</style>
"""

# Streamlit application
st.title("Brain Stroke Prediction")

# Apply CSS styles using st.markdown
st.markdown(css_styles, unsafe_allow_html=True)

def main():
    st.header("Predict Patient Condition")

    with st.form(key='prediction_form'):
        gender = st.number_input('Gender (Male:0 Female:1)', min_value=0, max_value=1)
        age = st.number_input('Age', min_value=0.0)
        hypertension = st.number_input('Hypertension (No:0 or Yes:1)', min_value=0, max_value=1)
        heart_disease = st.number_input('Heart Disease (No:0 or Yes:1)', min_value=0, max_value=1)
        work_type = st.number_input('Work Type (Private:0 Self Employed:1, Govt_job:2, Children:3)', min_value=0, max_value=3)
        Residence_type = st.number_input('Residence Type (Urban:0, Rural:1)', min_value=0, max_value=1)
        avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, step=0.01)
        bmi = st.number_input('BMI', min_value=0.0, step=0.01)
        smoking_status = st.number_input('Smoking Status (Neversmoked:0, Formerly Smoked:1, Smokes:2)', min_value=0, max_value=2)

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            input_data = np.array([[gender, age, hypertension, heart_disease, work_type, Residence_type,
                                    avg_glucose_level, bmi, smoking_status]])
            prediction = model.predict_proba(input_data)[:, 1][0]  # Get the probability for the positive class

            if prediction >= 0.70:
                result = "Advice for check up"
            elif 0.40 <= prediction <= 0.69:
                result = "Be wary"
            else:
                result = "No"

            # Store the prediction in the database
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('''INSERT INTO predictions (gender, age, hypertension, heart_disease, work_type, Residence_type,
                         avg_glucose_level, bmi, smoking_status, prediction, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (gender, age, hypertension, heart_disease, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, prediction, result))
            conn.commit()
            conn.close()

            st.success(f'Prediction: {prediction * 100:.2f}%, Result: {result}')
            
            st.markdown('<a href="/" style="text-decoration:none;"><button type="button">Submit Another Value</button></a>', unsafe_allow_html=True)

            st.write("Was the prediction correct?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Yes'):
                    st.write(f"Feedback received: Yes")
                    st.success('Thank you for your feedback!')
                    # Code to store feedback in the database can go here

            with col2:
                if st.button('No'):
                    st.write(f"Feedback received: No")
                    st.success('Thank you for your feedback!')
                    # Code to store feedback in the database can go here

    st.write("""
        <p>Patients with a percentage near 50% or over should consider having a health checkup.</p>
        <p>Patients above 70% should visit the hospital for potential brain stroke or stroke-related health concerns.</p>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
