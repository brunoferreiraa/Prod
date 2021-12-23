'''
Script for ARIMA algorithm, with univariate time-series data

Non-seasonal ARIMA(p,d,q)

p is the number of autoregressive terms,
d is the number of nonseasonal differences needed for stationarity, and
q is the number of lagged forecast errors in the prediction equation.

Steps :

1. Find how to stationarize the time serie (I) in ARIMA part -> d
2. Find the auto-regressive (AR) in ARIMA part -> p
3. Find the Moving-Average (MA) in ARIMA part -> q
4. Fit
5. Assess performance

'''
# Useful link: https://people.duke.edu/~rnau/411arim.htm


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf

# Variables (to be modified)
csv_path = "data.csv"
output_name = ""
model_pickle_path = "ml_pickles/test.pkl"

# Open csv
data = pd.read_csv(csv_path)

# Assuming data is a univariate time-series:
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# Plot 
data.plot()
plt.title("Some time-series")
plt.show()

# Train Test Split for time-series, 80% train, 20% test
split = 0.8 
data_split = int(len(data)*split)
data_train = data[:data_split]
data_test = data[data_split:]


# Check if there is seasonality or trend - ARIMA or SARIMA?
def check_seasonality_trend(data):
    result_add = seasonal_decompose(data, model='additive')
    result_add.plot()
    plt.title("Additive Model")
    
    result_mul = seasonal_decompose(data, model='multiplicative')
    result_mul.plot()
    plt.title("Multiplicative Model")

    plt.show()

# Ensure Stationarity
def find_d(y):
    
    '''Find the minimum order of differencing we need to make it stationnary'''

    # Check stationarity using Dick Fuller test
    result = adfuller(y)
    pvalue = result[1]

    # Base Case First Diff
    n_diff = 1
    diff = pd.Series(y).diff()
    result = adfuller(diff[n_diff:])
    pvalue=result[1]
    n_diff += 1

    # plot acf
    plot_acf(diff[n_diff-1:], title="Autocorrelation {} order".format(n_diff-1))

    # p-value should be less than 0.05 to have a 95% confidence in the stationarity.
    # stationarized through differencing.
    while pvalue > 0.05:

        diff = pd.Series(diff).diff()
        result = adfuller(diff[n_diff:])
        pvalue=result[1]
        n_diff += 1

        # plot acf
        plot_acf(diff[n_diff-1:], title="Autocorrelation {} order".format(n_diff-1))
        plt.show()

    return n_diff-1

# Find Auto-Regressive
def find_p(y_diff):

    '''AR order (p) can be found by investigating the partial autocorrelation applied to y_diff'''

    plot_pacf(y_diff)
    plt.show()

# Find the Moving Average
def find_q(y_diff):

    '''MA order (q) can be found by looking at the autocorrelation plot applied to y_diff'''

    plot_acf(y_diff)
    plt.show()


# Model ARIMA
def build_model(data_train, p, d, q):

    arima=ARIMA(data_train, order=(p,d,q))
    arima=arima.fit()
    print(arima.summary())

    return arima

# Forecast
def predict_model(arima):

    forecast = arima.forecast(len(data_test), alpha=0.05)
    return(forecast)

# Plot Forecast
def plot_forecast(forecast, data_train, data_test):

    fc_series = pd.Series(forecast, index=data_test.index)

    # Plot
    plt.figure(figsize=(12,5), dpi=100)
    plt.plot(data_train, label='training')
    plt.plot(data_test, label='actual')
    plt.plot(fc_series, label='forecast')

    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


# Plot Residuals
def plot_residuals(arima):

    residuals = arima.resid
    fig, ax = plt.subplots(1,2, figsize=(17,5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

# Metrics
def forecast_accuracy(forecast, y_true):
    
    y_pred =  pd.Series(forecast, index=data_test.index)
    y_true =  pd.Series(y_true['x'], index=data_test.index)
    mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))  # Mean Absolute Percentage Error
    me = np.mean(y_pred - y_true)             # ME
    mae = np.mean(np.abs(y_pred - y_true))    # MAE
    rmse = np.mean((y_pred - y_true)**2)**.5  # RMSE
    return({'mape':mape, 'me':me, 'mae': mae, 'rmse':rmse})


# -------------------------------------------------------------------------------------------------------#

# Usual Flow for analysis - (To delete after)

check_seasonality_trend(data_train)

d = find_d(data['x'])

x = input("By the analysis, d should be {}. Do you want to keep the value (Please consider the hypothesis of being over-differentiating) - y/n ".format(d))
if x == "y":
    pass
else:
    value = input("What value for d? ")
    d = int(value)

y_diff = data['x'].diff(d)[d:]

find_p(y_diff)
p = int(input("After graph analysis which value for p? "))
if not isinstance(p, int):
    print("Please insert a valid value")
else:
    pass

find_q(y_diff)
q = int(input("After graph analysis which value for q? "))
if not isinstance(q, int):
    print("Please insert a valid value")
else:
    pass

arima = build_model(data_train, p, d, q)

forecast = predict_model(arima)

plot_forecast(forecast, data_train, data_test)

plot_residuals(arima)

# y_pred =  pd.Series(forecast, index=data_test.index)
# print(y_pred)
# print(data_test)
# print(type(y_pred))
# print(type(data_test))
print(forecast_accuracy(forecast, data_test))


