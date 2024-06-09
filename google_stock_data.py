import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Load and preprocess data
data = pd.read_csv('Google_train_data.csv')
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
data = data.dropna()
train_data = data.iloc[:, 4:5].values

# Scaling the data
sc = MinMaxScaler(feature_range=(0, 1))
train_data = sc.fit_transform(train_data)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0]) 
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# Plot training loss
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Load test data
test_data = pd.read_csv('Google_test_data.csv')
test_data["Close"] = pd.to_numeric(test_data["Close"], errors='coerce')
test_data = test_data.dropna()
test_data = test_data.iloc[:, 4:5]
y_test = test_data.iloc[60:, 0:].values

# Preparing test data
input_closing = test_data.iloc[:, 0:].values 
input_closing_scaled = sc.transform(input_closing)
X_test = []
for i in range(60, len(test_data)):
    X_test.append(input_closing_scaled[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions
y_pred = model.predict(X_test)
predicted_price = sc.inverse_transform(y_pred)

# Plotting the results
plt.plot(y_test, color='red', label='Actual Stock Price')
plt.plot(predicted_price, color='green', label='Predicted Stock Price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
