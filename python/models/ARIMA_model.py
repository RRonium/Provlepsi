import pandas as pd
import math
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

df = pd.read_csv("")

#These are the variables who's values are to be aquired by the front end
response_variable = ""
split_fraction = None
p = None
d = None
q = None
forecast_steps = None

#ARIMA is univariate: it only looks at the history of the response itself
series = df[response_variable].astype(float).reset_index(drop=True)

#Manual split as Time-Series data, can't use train_test_split due to its randomness
split_index = int(len(series) * (1 - split_fraction))
y_train, y_test = series.iloc[:split_index], series.iloc[split_index:]

arima_model = ARIMA(y_train, order=(p, d, q)).fit()

#Error Eval: forecast the whole held-out window in one go
sample_predictions = arima_model.forecast(steps=len(y_test)).to_numpy()
MAE = mean_absolute_error(y_test, sample_predictions)
MSE = mean_squared_error(y_test, sample_predictions)
RMSE = math.sqrt(MSE)
AIC = float(arima_model.aic)
BIC = float(arima_model.bic)

#User's Prediction: refit on the full series and forecast past the last observation
user_entry = {}
steps = int(user_entry.get("steps", forecast_steps))
full_model = ARIMA(series, order=(p, d, q)).fit()
forecast = [float(v) for v in full_model.forecast(steps=steps)]
single_prediction = forecast[0]
