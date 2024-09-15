import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.python.keras.saving.saved_model_experimental import sequential

#load teh dataset
df = pd.read_csv('AAPL.csv')

#Set the datetime index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)

#Resample the data to daily frequency
df = df.resample('D').mean()

#Createe the new column for target variable(future prices)
df['Future_Price'] = df['Close'].shift(-1)

#Drop the last row (since it doesn't have future value(NaN))
df = df.dropna()

X = df.drop(['Future_Price'], axis = 1)
y = df['Future_Price']

#Split the data in to testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Normalisation
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)

#Create a LSTM model
model = Sequential([
    LSTM(units = 50,
         return_sequences = True,
         input_shape = (X_train.shape[1], 1)),

    Dense(1)
])

#Compiling the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#model training
model.fit(X_train, y_train, epochs = 10, batch_size = 1, verbose = 2)

#mAke predictions
Predictions = model.predict(X_test)

# Make predictions
Predictions = model.predict(X_test)

# Reshape Predictions to 2D array (n_samples, n_features)
Predictions = Predictions.reshape(-1, Predictions.shape[-1])

# Calculate MAE
MAE = np.mean(np.abs(Predictions - y_test.values))
print('MAE : ', MAE)

# **Predicting Future Stock Prices**

# Prepare the data for future predictions
future_days = 30  # number of days to predict
future_data = scaler.transform(df.tail(future_days).drop('Future_Price', axis = 1))

#reshape the data
future_data = future_data.reshape(-1, future_days,1)


# Make predictions on the future data
future_predictions = model.predict(future_data)

# Create a date range for the predicted prices
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')

# Print the predicted future prices
print('Predicted Future Prices:')
print(pd.DataFrame({'Date': future_dates.date, 'Price': future_predictions[0].flatten()}).to_string(index=False))