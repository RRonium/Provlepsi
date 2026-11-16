import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("")

#Will have to be aquired from frontend
response_variable = ""
split_fraction = None 

X = df.drop(columns=[response_variable])
y = df[response_variable] #will have to be provided by the user

#manual train test split as can't use the function from sklearn as it'll jumble up and we can't do that with time-series data
split_index = int(len(df)*(1-split_fraction))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

#fitting model
lm = LinearRegression()
lm.fit(X_train,y_train)

#error Evaluation
sample_predictions = lm.predict(X_test)
MAE = mean_absolute_error(sample_predictions,y_test)
MSE = mean_squared_error(sample_predictions,y_test)
RMSE = math.sqrt(MSE)

#final user prediction to be received in the form of JSON and converted to pandas dataframe
user_entry = {} 
user_df = pd.DataFrame([user_entry])[X.columns]
single_prediction = lm.predict(user_df)[0]