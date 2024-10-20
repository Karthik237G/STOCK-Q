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
