import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *

# Load training data and split between dependent and independent variables
train_data_df = pd.read_csv('model/training_scaled.csv')
X = train_data_df.drop(['gender', 'age'], axis=1).values
Y = train_data_df[['gender', 'age']].values

# Load testing data and split between dependent and independent variables
test_data_df = pd.read_csv("model/test_scaled.csv")
X_test = test_data_df.drop(['gender', 'age'], axis=1).values
Y_test = test_data_df[['gender', 'age']].values

# Define the model
model = Sequential()
model.add(Dense(150, input_dim=len(train_data_df.columns) - 2, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Create TensorBoard logs
tensorboard = keras.callbacks.TensorBoard(log_dir="model/logs/", histogram_freq=1)

# Train the model
model.fit(X, Y, epochs=100, shuffle=True, callbacks=[tensorboard])

# Evaluate model using testing data
test_error_rate = model.evaluate(X_test, Y_test, verbose=0) * 100
print("MSE for test dataset: {0:.2f}%".format(test_error_rate))

# Export the model
model.save("model/trained_model.h5")