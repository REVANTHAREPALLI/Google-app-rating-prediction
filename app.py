import streamlit as st
import pickle
import numpy as np

# Load the model and PCA
model = pickle.load(open('model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

st.title("Google Play Store App Success Predictor")

# Input form
app_name = st.text_input("App Name (Not used in prediction)")
category = st.selectbox("Category", ['FAMILY', 'GAME', 'TOOLS', 'EDUCATION', 'PRODUCTIVITY'])  # Update list as needed
size = st.number_input("Size of the app in MB", min_value=1.0)
installs = st.number_input("Number of installs from Play Store", min_value=0)
paid_status = st.selectbox("Is the app paid or not", ['Free', 'Paid'])
price = st.number_input("Price of the app (if free, type 0.0)", min_value=0.0)
rating = st.number_input("PG rating for the app (0 to 5)", min_value=0.0, max_value=5.0)
year = st.number_input("Year it was released", min_value=2000, max_value=2025)
month = st.number_input("Month it was released (1-12)", min_value=1, max_value=12)
day = st.number_input("Day of the month it was released", min_value=1, max_value=31)

# Encode input
paid = 0 if paid_status == 'Free' else 1

# Create input vector (adjust according to training order)
input_data = np.array([[size, installs, paid, price, rating, year, month, day]])

# Apply PCA
input_pca = pca.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_pca)
    st.success(f"Predicted Success Score / Category: {prediction[0]}")
