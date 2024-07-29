import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib

housing = pd.read_csv('housing.csv')

x_train, x_test, y_train, y_test = train_test_split(housing.median_income, housing.median_house_value, test_size = 0.2)

regr = LinearRegression()
regr.fit(np.array(x_train).reshape(-1,1), y_train)

print("training successfully completed")
joblib.dump(regr,"linear_regrssion.joblib")
