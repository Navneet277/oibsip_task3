# oibsip_task3

# Task 3: 

Car Price Prediction using Machine Learning

# Task Overview:
--> This project predicts the selling price of used cars based on various features such as mileage, fuel type, car age, and more. It utilizes a Linear Regression 
    model, trained on a dataset of used cars, to generate price predictions.

# Project Structure:

car-price-prediction/

│
├── data/

│   └── car_data.csv                # Dataset used for training

├── outputs                         # Outputs(Screen)

│   └── choice_1.png  

│   └── choice_2.png  


├── src                             # Main Python script (training + prediction)

│   └── car_prediction_jupyter.ipynb 

│   └── car_prediction_py.py  

└── requirements.txt                # Dependencies for the project

# Features Used in the Model:

The model uses the following features for prediction:

Present_Price: The current price of the car (in lakhs).

Kms_Driven: Total kilometers driven by the car.

Owner: Number of previous owners (0 for no previous owner, 1 for one owner, etc.).

Age: The age of the car (calculated as 2023 - Year of Purchase).

Fuel_Type: Whether the car runs on Petrol or Diesel.

Seller_Type: Dealer or Individual seller.

Transmission: Manual or Automatic transmission.

# Steps to Run the Project:

1. Clone the Repository

Clone this project to your local machine:

git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction

2. Install Dependencies

Install the required Python libraries using the following command:

pip install -r requirements.txt

3. Dataset

Ensure the dataset (car_data.csv) is located in the data/ folder. If you don't have the dataset, upload it or modify the path in main.py.

4. Train the Model

Run the following command to train the model:
python main.py

*When prompted*:

*Select 1* to train the model.

The trained model will be saved as models/linear_regression_model.pkl.

5. Predict Car Prices

Run the same main.py script to predict car prices:
python main.py

*When prompted*:

*Select 2* for car price prediction.

Enter details such as year of purchase, present price, mileage, fuel type, etc., as prompted.

The predicted car price will be displayed.

# Example Usage:

# Training the Model

Run the script:
python main.py

*Select 1* to train the model. 

Example output:

Training the model...

Model trained and saved at: models/linear_regression_model.pkl

# Predicting Car Prices

Run the script:
python main.py

*Select 2* to predict car prices. Enter details as prompted, 

for example:

Enter the year of purchase (e.g., 2015): 2018

Enter the present price (in lakhs, e.g., 6.5): 5.5

Enter the kilometers driven (e.g., 50000): 30000

Enter the fuel type (Petrol/Diesel): Petrol

Enter the seller type (Dealer/Individual): Dealer

Enter the transmission type (Manual/Automatic): Manual

Enter the number of previous owners (e.g., 0): 0

Estimated Selling Price: ₹3.15 Lakhs

# Dataset:

The dataset used in this project contains the following columns:

Car_Name: The name of the car.

Year: The year the car was purchased.

Present_Price: The current ex-showroom price of the car.

Kms_Driven: Total kilometers driven by the car.

Fuel_Type: Type of fuel (Petrol/Diesel).

Seller_Type: Whether the car is being sold by a Dealer or an Individual.

Transmission: Type of transmission (Manual/Automatic).

Owner: Number of previous owners.

Selling_Price: The price at which the car was sold (target variable).

# Technologies Used:

Programming Language: 

Python

Libraries:

pandas for data manipulation.

numpy for numerical operations.

scikit-learn for building the regression model.

pickle for saving and loading the trained model.

# Prerequisites:

Before running the project, ensure you have the following installed:

Python 3.8 or above

Libraries listed in requirements.txt (install using pip install -r requirements.txt).

# Future Enhancements:

(1) Add support for additional regression models such as Random Forest or Gradient Boosting.

(2) Create a user-friendly web interface using Flask or Streamlit.

(3) Expand the dataset with more features and records for better accuracy.
