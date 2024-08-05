import optuna
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib

# Load the dataset
housing = pd.read_csv('housing.csv')

# Define the objective function
def objective(trial):
    # Suggest hyperparameters for Optuna to try
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(housing[['median_income']], housing['median_house_value'], test_size=0.2, random_state=42)
    
    # Normalize the data manually if needed
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Create and train the model
    regr = LinearRegression(fit_intercept=fit_intercept)
    regr.fit(x_train, y_train)
    
    # Predict and calculate MSE
    y_pred = regr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # You can change the number of trials as needed

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)
print("Best MSE: ", study.best_value)

# Train the final model using the best hyperparameters
best_params = study.best_params
x_train, x_test, y_train, y_test = train_test_split(housing[['median_income']], housing['median_house_value'], test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
regr = LinearRegression(fit_intercept=best_params['fit_intercept'])
regr.fit(x_train, y_train)

# Save the final model
joblib.dump(regr, "linear_regression_best.joblib")
print("Training with the best hyperparameters completed and model saved")
