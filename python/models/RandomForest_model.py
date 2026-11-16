import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("")

#These are the variables who's values are to be aquired by the front end
response_variable = "" 
split_fraction = None
n_estimators = None
max_depth = None 
min_samples_split = None

#We will have to create some lag so that the RandomForestRegressor model can see the trend of data
#We do this as it's Time-Series data (mentioned by sir)
df['lag1']=df[response_variable].shift(1)
df['lag2']=df[response_variable].shift(2)
df['lag3']=df[response_variable].shift(3)

df = df.dropna()

X = df.drop(columns=response_variable)
y=df[response_variable]

#Manual split as Time-Series data, can't use train_test_split due to its randomness
split_index=int(len(df)*(1-split_fraction))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

rf_model = RandomForestRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=42
)
rf_model.fit(X_train,y_train)

#Error Eval
sample_predictions = rf_model.predict(X_test)
MAE = mean_absolute_error(y_test, sample_predictions)
MSE = mean_squared_error(y_test, sample_predictions)
RMSE = math.sqrt(MSE)

#User's Prediction
user_entry = {} 
user_df = pd.DataFrame([user_entry])[X.columns]
single_prediction = rf_model.predict(user_df)[0]