import streamlit as st
import pickle
import numpy as np

# Load the trained model and PCA
model = pickle.load(open('model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))

st.title("Google Play Store App Success Predictor")

# Input form
app_name = st.text_input("App Name (Not used in prediction)")

# Add 'Category' as the missing 9th feature
category = st.selectbox("Category", [
    'FAMILY', 'GAME', 'TOOLS', 'EDUCATION', 'PRODUCTIVITY'
])  # Modify this list as per your training dataset

# Map category to integer (you must use the same mapping used during training!)
category_map = {
    'FAMILY': 0,
    'GAME': 1,
    'TOOLS': 2,
    'EDUCATION': 3,
    'PRODUCTIVITY': 4
}
category_encoded = category_map[category]

# Other features
size = st.number_input("Size of the app in MB", min_value=1.0)
installs = st.number_input("Number of installs", min_value=0)
paid_status = st.selectbox("Is the app paid or not", ['Free', 'Paid'])
price = st.number_input("Price of the app (₹)", min_value=0.0)
rating = st.number_input("App Rating (0.0 to 5.0)", min_value=0.0, max_value=5.0)
year = st.number_input("Year of release", min_value=2000, max_value=2030)
month = st.number_input("Month of release", min_value=1, max_value=12)
day = st.number_input("Day of release", min_value=1, max_value=31)

# Encode 'Paid' as 0/1
paid = 0 if paid_status == 'Free' else 1

# Create full input array (9 features now)
input_data = np.array([[category_encoded, size, installs, paid, price, rating, year, month, day]])

# Apply PCA
input_pca = pca.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_pca)
    st.success(f"Predicted Success Category / Rating: {prediction[0]}")
