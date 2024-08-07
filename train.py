import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import mlflow
from mlflow.models import infer_signature

# Set up MLflow configuration
mlflow.set_tracking_uri('http://localhost:5000')
# Load the dataset
housing = pd.read_csv('housing.csv')

# Define the objective function
def objective(trial):
    # Suggest hyperparameters for Optuna to try
    model_type = trial.suggest_categorical('model_type', ['ridge', 'lasso'])
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    alpha = trial.suggest_float('alpha', 1e-5, 1e2, log=True)  # Regularization strength
    
    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(housing[['median_income']], housing['median_house_value'], test_size=0.2, random_state=42)
    
    # Normalize the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    signature = infer_signature(x_train, y_train)

    # Create and train the model with MLFLow Tracking
    with mlflow.start_run(nested=True):
        if model_type == 'ridge':
            regr = Ridge(fit_intercept=fit_intercept, alpha=alpha)
        else:
            regr = Lasso(fit_intercept=fit_intercept, alpha=alpha)
        
        regr.fit(x_train, y_train)
        
        # Predict and calculate MSE
        y_pred = regr.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)

        mlflow.log_params({"fit_intercept":fit_intercept})
        mlflow.log_params({"model_type":model_type})
        mlflow.log_params({"alpha":alpha})
        mlflow.log_metric("eval_rmse", mse)
        mlflow.sklearn.log_model(regr,"model",signature=signature)                     
        
        return mse
    

mlflow.set_experiment("Housing experiment 5")
with mlflow.start_run():

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # You can change the number of trials as needed

    # Log the best parameters, loss, and model
    mlflow.log_params(study.best_params)
    mlflow.log_metric("eval_rmse", study.best_value)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    print("Best MSE: ", study.best_value)

    # Train the final model using the best hyperparameters
    best_params = study.best_params
    x_train, x_test, y_train, y_test = train_test_split(housing[['median_income']], housing['median_house_value'], test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    if best_params['model_type'] == 'ridge':
        regr = Ridge(fit_intercept=best_params['fit_intercept'], alpha=best_params['alpha'])
    else:
        regr = Lasso(fit_intercept=best_params['fit_intercept'], alpha=best_params['alpha'])

    regr.fit(x_train, y_train)

    signature = infer_signature(x_train, y_train)

    mlflow.sklearn.log_model(regr, "model",signature=signature)

# Save the final model
joblib.dump(regr, "linear_regression_best.joblib")
print("Training with the best hyperparameters completed and model saved")
