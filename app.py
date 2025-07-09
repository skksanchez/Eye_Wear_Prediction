import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
brand_model = joblib.load('brand_model.pkl')
frame_model = joblib.load('frame_model.pkl')
brand_le = joblib.load('brand_encoder.pkl')
frame_le = joblib.load('frame_encoder.pkl')

# Title
st.title("Optical Brand & Frame Shape Predictor")

# Input form
with st.form("prediction_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", options=['M', 'F'])
    address = st.text_input("Address")

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create input dataframe
    input_df = pd.DataFrame([[age, gender.strip(), address.strip()]],
                            columns=['Age', 'Gender', 'Address'])

    # Predict brand and frame
    pred_brand = brand_le.inverse_transform(brand_model.predict(input_df))[0]
    pred_frame = frame_le.inverse_transform(frame_model.predict(input_df))[0]

    # Display result
    st.success(f"👤 {name} ({gender}) from {address}")
    st.markdown(f"### 🕶️ Predicted Brand: `{pred_brand}`")
    st.markdown(f"### 🔲 Predicted Frame Shape: `{pred_frame}`")
