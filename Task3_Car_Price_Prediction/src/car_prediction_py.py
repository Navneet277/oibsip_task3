"""
Original file is located at
    https://colab.research.google.com/drive/1EJrJ-EXcgwWuYljwZcE0dGZcclIRR9R6
"""

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Step 1: Data Preprocessing
def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df.drop(['Car_Name'], axis=1, inplace=True)
    df['Age'] = 2023 - df['Year']
    df.drop(['Year'], axis=1, inplace=True)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Step 2: Train the Model
def train_model(data_path, model_save_path):
    df = preprocess_data(data_path)
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model trained successfully! Mean Squared Error: {mse}")

# Step 3: Prediction Interface
def predict_price(model_path):
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please train the model first.")
        return
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    print("\n--- Car Price Prediction ---")
    year = int(input("Enter the year of purchase (e.g., 2015): "))
    present_price = float(input("Enter the present price (in lakhs, e.g., 6.5): "))
    kms_driven = int(input("Enter the kilometers driven (e.g., 50000): "))
    fuel_type = input("Enter the fuel type (Petrol/Diesel): ")
    seller_type = input("Enter the seller type (Dealer/Individual): ")
    transmission = input("Enter the transmission type (Manual/Automatic): ")
    owner = int(input("Enter the number of previous owners (e.g., 0): "))

    # Prepare the input data
    age = 2023 - year
    fuel_petrol = 1 if fuel_type == "Petrol" else 0
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    seller_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    input_data = pd.DataFrame([{
        'Present_Price': present_price,
        'Driven_kms': kms_driven,  # Changed Kms_Driven to Driven_kms
        'Owner': owner,
        'Age': age,
        'Fuel_Type_Diesel': fuel_diesel,
        'Fuel_Type_Petrol': fuel_petrol,
        'Selling_type_Individual': seller_individual,  # Changed Seller_Type_Individual to Selling_type_Individual
        'Transmission_Manual': transmission_manual
    }])


    # Predict the price
    prediction = model.predict(input_data)[0]
    print(f"\nEstimated Selling Price: ₹{prediction:.2f} Lakhs")

# Main Function
if __name__ == "__main__":
    data_path = "data/car data (1).csv"
    model_path = "models/linear_regression_model.pkl"

    print("1. Train the Model")
    print("2. Predict Car Price")
    choice = int(input("Enter your choice (1/2): "))

    if choice == 1:
        train_model(data_path, model_path)
    elif choice == 2:
        predict_price(model_path)
    else:
        print("Invalid choice!")
