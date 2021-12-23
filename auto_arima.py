'''AUTO ARIMA - Like Gridsearch'''

import pandas as pd
import matplotlib.pyplot as plt

# Variables (to be modified)
csv_path = "data.csv"
output_name = ""
model_pickle_path = "ml_pickles/test.pkl"

# Open csv
data = pd.read_csv(csv_path)

# Assuming data is a univariate time-series:
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

import pmdarima as pm
model = pm.auto_arima(data, test='adf',
                      start_p=0, max_p=3,
                      start_q=0, max_q=3,
                      m=1,
                      trace=True, 
                      error_action='ignore', 
                      suppress_warnings=True,)
print(model.summary())

model.plot_diagnostics(figsize=(17,9))
plt.show()