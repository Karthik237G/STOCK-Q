import pandas as pd
import numpy as np
import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
import yfinance as yf

plt.style.use('fivethirtyeight')
import pandas_datareader.data as web

# Fetching data for Apple Inc. (AAPL) from Yahoo Finance
ticker = yf.Ticker('AAPL')
df=ticker.history(period='max')
# Define the window size for SMA, e.g., 20 days
window_size = 20

df['SMA'] = df['Close'].rolling(window=window_size).mean()


# Calculate the smoothing factor
smoothing_factor = 2 / (window_size + 1)

# Initialize the EMA column with NaN values
df['EMA'] = None

# Calculate EMA using the provided formula with .loc to avoid SettingWithCopyWarning
for i in range(len(df)):
    if i == 0:
        # Use the first closing price as the initial EMA value
        df.loc[df.index[i], 'EMA'] = df['Close'].iloc[i]
    else:
        # Calculate EMA based on previous EMA and current price
        df.loc[df.index[i], 'EMA'] = (
            (df['Close'].iloc[i] * smoothing_factor) +
            (df['EMA'].iloc[i - 1] * (1 - smoothing_factor))
        )
df.tail()

rsi_period = 14


delta = df['Close'].diff()

# Separate gains and losses
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()

# Calculate Relative Strength (RS)
rs = gain / loss

# Calculate RSI
df['RSI'] = 100 - (100 / (1 + rs))

# Display the first few rows of the dataframe with Close and RSI
df.tail()

# Calculate True Range (TR)
df['High-Low'] = df['High'] - df['Low']
df['High-PC'] = abs(df['High'] - df['Close'].shift(1))
df['Low-PC'] = abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['High-Low', 'High-PC', 'Low-PC']].max(axis=1)

# Calculate Directional Movement (+DM and -DM)
df['+DM'] = df['High'].diff()
df['-DM'] = -df['Low'].diff()
df['+DM'] = df['+DM'].where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), 0)
df['-DM'] = df['-DM'].where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), 0)

# Calculate Smoothed Averages
period = 14
df['ATR'] = df['TR'].rolling(window=period).mean()
df['+DI'] = 100 * (df['+DM'].rolling(window=period).mean() / df['ATR'])
df ['-DI'] = 100 * (df ['-DM'].rolling(window=period).mean() / df ['ATR'])

# Calculate ADX
df ['DX'] = 100 * (abs(df ['+DI'] - df ['-DI']) / (df ['+DI'] + df ['-DI']))
df['ADX '] = df ['DX'].rolling(window=period).mean()

df.tail()

