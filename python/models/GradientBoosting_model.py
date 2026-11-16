import pandas as pd
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("")

#These are the variables who's values are to be aquired by the front end
response_variable = ""
split_fraction = None
n_estimators = None
learning_rate = None
max_depth = None
subsample = None
max_lags = None

#Same idea as the Random Forest model: autoregressive lags of the response
#so the boosted trees can see the recent history of the series
for lag in range(1, max_lags + 1):
    df[f"lag{lag}"] = df[response_variable].shift(lag)

df = df.dropna()

X = df.drop(columns=response_variable)
y = df[response_variable]

#Manual split as Time-Series data, can't use train_test_split due to its randomness
split_index = int(len(df) * (1 - split_fraction))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

gb_model = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    random_state=42
)
gb_model.fit(X_train, y_train)

#Error Eval
sample_predictions = gb_model.predict(X_test)
MAE = mean_absolute_error(y_test, sample_predictions)
MSE = mean_squared_error(y_test, sample_predictions)
RMSE = math.sqrt(MSE)

#User's Prediction
user_entry = {}
user_df = pd.DataFrame([user_entry])[X.columns]
single_prediction = float(gb_model.predict(user_df)[0])

#Which inputs the ensemble leaned on most
feature_weights = dict(zip(X.columns, gb_model.feature_importances_))
