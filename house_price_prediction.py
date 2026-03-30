import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# Load Dataset
df = pd.read_csv("housing_price_dataset.csv")

print("Dataset Preview:")
print(df.head())


# Convert Text Column
label = LabelEncoder()
df['Neighborhood'] = label.fit_transform(df['Neighborhood'])


# Features and Target
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
y = df['Price']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)


# Create Model
model = LinearRegression()


# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed")


# Predict Test Data
y_pred = model.predict(X_test)

print("\nPredicted Prices:")
print(y_pred)


# Model Accuracy
print("\nModel Accuracy")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Predict New House
new_house = pd.DataFrame({
    'SquareFeet': [2000],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Neighborhood': [1],
    'YearBuilt': [2015]
})

price = model.predict(new_house)

print("\nPredicted New House Price:", price[0])


# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


# Feature Importance
print("\nModel Coefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)