import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model_filename = 'ngp_rest.pkl'
model = joblib.load(model_filename)

# Load the dataset
db = pd.read_csv("Swiggy_Nagpur.csv")

# Create a label encoder for category encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
db['Category_encoded'] = label_encoder.fit_transform(db['Category'])

# Streamlit app
st.title("Nagpur Restaurant Rating Predictor")

# Create a dropdown list for food categories
food_categories = db['Category'].unique()
user_input_category = st.selectbox("Select a food category:", food_categories)

# Encode user input category using the label encoder
user_input_encoded = label_encoder.transform([user_input_category])

# Predict ratings for all restaurants and store them in a new column
db['Predicted_Rating'] = model.predict(db['Category_encoded'].values.reshape(-1, 1))

# Filter restaurants based on the selected category
filtered_db = db[db['Category_encoded'] == user_input_encoded[0]]

# Sort restaurants by predicted rating in descending order
sorted_db = filtered_db.sort_values(by='Predicted_Rating', ascending=False)

st.subheader("Results:")
st.write("Top-rated restaurants in the category '{}':".format(user_input_category))
st.table(sorted_db[['Restaurant Name', 'Rating', 'Predicted_Rating']])
