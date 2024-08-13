import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv("E:\\car data.csv")
data.info()
print(data)


# Separate features and target variable
X = data.drop(columns=['Selling_Price'])
y = data['Selling_Price']

# Define categorical and numerical features
categorical_features = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Preprocessing: Scaling numerical features and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model pipeline 
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best R2 score: ", grid_search.best_score_)



# Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)



# Function to predict car price based on user input
def predict_car_price(model):
    # Gather user input
    car_name = input("Enter Car Name: ")
    year = int(input("Enter Year: "))
    present_price = float(input("Enter Present Price: "))
    driven_kms = float(input("Enter Driven KMs: "))
    fuel_type = input("Enter Fuel Type (Petrol/Diesel/CNG): ")
    selling_type = input("Enter Selling Type: ")
    transmission = input("Enter Transmission (Manual/Automatic): ")
    owner = int(input("Enter Number of Owners: "))

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Car_Name': [car_name],
        'Year': [year],
        'Present_Price': [present_price],
        'Driven_kms': [driven_kms],
        'Fuel_Type': [fuel_type],
        'Selling_type': [selling_type],
        'Transmission': [transmission],
        'Owner': [owner]
    })

    # Predict the price using the trained model
    predicted_price = model.predict(input_data)
    print(f"The predicted selling price for the car is: {predicted_price[0]}")

# Call the function to predict car price based on user input
predict_car_price(best_model)