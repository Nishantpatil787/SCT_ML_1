import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# Load Dataset
df = pd.read_csv("housing_price_dataset.csv")

# Encode Neighborhood
label = LabelEncoder()
df['Neighborhood'] = label.fit_transform(df['Neighborhood'])


# Features
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
y = df['Price']


# Train Model
model = LinearRegression()
model.fit(X, y)


# UI Title
st.title("🏠 House Price Prediction System")
st.write("Enter House Details")


# User Input
square_feet = st.number_input("Square Feet", 500, 5000, 2000)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 5, 2)

neighborhood = st.selectbox(
    "Neighborhood",
    ["Urban", "Suburb", "Rural"]
)

year_built = st.number_input("Year Built", 1950, 2025, 2015)


# Convert Neighborhood
neighborhood_map = {
    "Urban": 0,
    "Suburb": 1,
    "Rural": 2
}

neighborhood_value = neighborhood_map[neighborhood]


# Predict Button
if st.button("Predict Price"):

    new_house = pd.DataFrame({
        'SquareFeet': [square_feet],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Neighborhood': [neighborhood_value],
        'YearBuilt': [year_built]
    })

    price = model.predict(new_house)

    # Convert to Rupees
    price_rupees = price[0]

    st.success(f"🏡 Predicted House Price: ₹ {price_rupees:,.2f}")


    # Graph UI
    st.subheader("📊 Price vs Square Feet")

    sample = df.sample(500)

    fig, ax = plt.subplots()
    ax.scatter(sample['SquareFeet'], sample['Price'])
    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Price (₹)")
    ax.set_title("House Price Distribution")

    st.pyplot(fig)


# Dataset Preview
if st.checkbox("Show Dataset"):

    st.subheader("Dataset Preview")
    st.write(df.head())