import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train.ml_selection import ml_selection
import pickle

# Variables (to be modified)
csv_path = ""
output_name = ""
model_pickle_path = "ml_pickles/test.pkl"

# ---------------------------------------
# FROM NOW ON PLEASE DONT MODIFY THE CODE
# FROM NOW ON PLEASE DONT MODIFY THE CODE
# FROM NOW ON PLEASE DONT MODIFY THE CODE
# ---------------------------------------

# Open csv
data = pd.read_csv(csv_path)

# Mode selection (trainer or predictor)
while True:
    try:
        mode = input("Enter mode ('train' or 'predict'):")
    except ValueError:
        print("Please Enter a valid mode ('train' or 'predict').")
        continue
    else:
        if mode != "train" and mode != "predict":
            break
        else:
            print("Invalid mode!")

# TRAIN ---------------------------------------------
if mode.lower == "train":
    # Splitting
    y = data.temp
    x = data.drop(output_name, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # TODO - Develop feature selection (only if necessary)
    # TODO - Develop feature selection (only if necessary)
    # TODO - Develop feature selection (only if necessary)

    # Select & optimize models + create a model pickle
    best_model_name = ml_selection(x_train, x_test, y_train, y_test)

    # Load pickle
    pickled_model = pickle.load(open(model_pickle_path, 'rb'))

    # Prediction
    predictions = pickled_model.predict(x_test)


elif mode.lower == "predict":
    # Splitting
    y_test = data.temp
    x_test = data.drop(output_name, axis=1)

    # Load pickle
    pickled_model = pickle.load(open(model_pickle_path, 'rb'))

    # Prediction
    predictions = pickled_model.predict(x_test)

# Check results from train or prediction
plt.scatter(y_test, predictions)
