import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

# Load the trained model
lg = pickle.load(open('placement.pkl', 'rb'))

# Web app interface
img = Image.open("data.jpg")
st.image(img, width=650)
st.title("Job Placement Prediction Model")

# User input fields
ssc_percentage = st.text_input("SSC Percentage")
hsc_percentage = st.text_input("HSC Percentage")
degree_percentage = st.text_input("GRADUATION PERCENTAGE")
mba_percent = st.text_input("POST GRADUATION PERCENTAGE")
work_experience = st.text_input("Work Experience")
gender_M = st.text_input("Gender (Enter 1 for Male, 0 for Female)")

# Submit button
if st.button("Submit"):
    if ssc_percentage and hsc_percentage and degree_percentage and mba_percent and work_experience and gender_M:
        try:
            # Convert input values to float and store in a NumPy array
            input_features = np.array([
                float(ssc_percentage),
                float(hsc_percentage),
                float(degree_percentage),
                float(mba_percent),
                float(work_experience),
                float(gender_M)  # Gender encoded as 1 (Male) or 0 (Female)
            ])

            # Reshape input data to match model input format
            np_df = input_features.reshape(1, -1)

            # Make prediction
            prediction = lg.predict(np_df)

            # Display result
            if prediction[0] == 1:
                st.success("✅ This Person Is Placed")
            else:
                st.error("❌ This Person is not Placed")

        except ValueError:
            st.warning("⚠️ Please enter valid numeric values for all fields.")

    else:
        st.warning("⚠️ Please fill in all fields to get a prediction.")
