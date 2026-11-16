import pandas as pd
import numpy as np
import math
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("")

# These are the variables who's values are to be aquired by the front end
response_variable = "" 
split_fraction = None 
alpha = 1.0
max_lags = None

feature_cols = [col for col in df.columns if col != response_variable]
for lag in range(1, max_lags + 1):
    for col in feature_cols:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)

df = df.dropna()

X = df.drop(columns=[response_variable])
y = df[response_variable]

# Manual split as Time-Series data, can't use train_test_split due to its randomness
split_index = int(len(df) * (1 - split_fraction))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

lm = Lasso(alpha=alpha, max_iter=10000)
lm.fit(X_train, y_train)

sample_predictions = lm.predict(X_test)

# Error Evaluation
MAE = mean_absolute_error(y_test, sample_predictions)
MSE = mean_squared_error(y_test, sample_predictions)
RMSE = math.sqrt(MSE)

# User's predicitons
user_entry = {} 
user_df = pd.DataFrame([user_entry])[X.columns]
single_prediction = float(lm.predict(user_df)[0])

# Feature Selection Tracking: Identifies which features Lasso kept vs zeroed out
feature_weights = dict(zip(X.columns, lm.coef_))
selected_features = [col for col, coef in feature_weights.items() if coef != 0]
zeroed_out_features = [col for col, coef in feature_weights.items() if coef == 0]