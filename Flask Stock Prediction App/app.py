import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess the dataset
df = pd.read_csv('AAPL.csv')  
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

close_data = df['Close'].values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data.reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        labels.append(data[i+seq_length])
    return np.array(sequences), np.array(labels)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=5)

predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predictions)

# Streamlit app for visualization
st.title("Stock Price Prediction with LSTM")
stock_symbol = st.selectbox("Choose a stock symbol:", ['AAPL', 'GOOG', 'AMZN'])

fig, ax = plt.subplots()
ax.plot(df['Date'][train_size + seq_length:], df['Close'][train_size + seq_length:], color='blue', label="Actual Price")
ax.plot(df['Date'][train_size + seq_length:], predicted_prices, color='red', label="Predicted Price")

ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)
