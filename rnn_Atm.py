# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('b9383file.csv')
training_set = dataset_train.iloc[:,2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(30, 367):
    X_train.append(training_set_scaled[i-30:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('b9383real.csv')
test_set = dataset_test.iloc[:,2:3].values
real_data = np.concatenate((training_set[0:367], test_set), axis = 0)

# Getting the predicted stock price of 2017
scaled_real_data = sc.fit_transform(real_data)
inputs = []
for i in range(367, 397):
    inputs.append(scaled_real_data[i-30:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_data = regressor.predict(inputs)
predicted_data = sc.inverse_transform(predicted_data)

# Visualising the results
plt.plot(real_data[367:], color = 'red', label = 'Real Cash Withdraw')
plt.plot(predicted_data, color = 'blue', label = 'Predicted Cash Withdraw')
plt.title('Cash withdraw Prediction')
plt.xlabel('Time')
plt.ylabel('Cash withdraw ')
plt.legend()
plt.show()