"""
Sean O'Brien
-feature2_LSTM continues to add LSTM predict functionality to our MACD. Once implimented,
will loop through MACD timeframe values in order to find optimal mix.
"""

"""The default values for the indicator are normally 12,26,9. In this case, the
MACD line itself is calculated as follows 12-period EMA – 26 period EMA. An important distinction though, I am using SMA, not EMA."""

"""The Difference Between EMA and SMA
The major difference between an EMA and an SMA is the sensitivity each one shows to changes in the data used in its calculation.

More specifically, the EMA gives higher weights to recent prices, while the SMA assigns equal weights to all values. The two averages are similar because they are interpreted in the same manner and are both commonly used by technical traders to smooth out price fluctuations. Since EMAs place a higher weighting on recent data than on older data, they are more responsive to the latest price changes than SMAs. That makes the results from EMAs more timely and explains why they are preferred by many traders."""

# Feature 3 will impliment the EMA

# Import Libraries, assign to reference variables
import yfinance as yf
import html5lib
import pandas as pd
import pandas_datareader
from pandas_datareader import data
import urllib.request, json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
from os import path
import datetime
import time


# Explicit type cast of variables
uInput1: str
uInput2: str
uInput3: str
tmp1: str
tmp2: str
tmp3: str
beginDate: str
endDate: str

shortTerm: int = 0
mediumTerm: int = 0
longTerm: int = 0
size: int = 0
shape: int = 0
trainData: int = 0
testData: int = 0
smoothingWindowSize: int = 0

highPrices: float = 0.0
midPrices: float = 0.0
lowPrice: float = 0.0

date_format = '%Y-%m-%d'

# User inputs
uInput1 = (input("Enter the ticker symbol: "))
tmp1 = uInput1.upper()
while(True):
    try:
        uInput2 = (input("Enter the begining date, format YYYY-MM-DD: "))
        tmp2 = uInput2
        date_obj = datetime.datetime.strptime(tmp2, date_format)
    except ValueError:
        print("Incorrect date format, should be YYYY-MM-DD")
    else:
        break
while(True):
    try:
        uInput3 = (input("Enter the ending date, format YYYY-MM-DD: "))
        tmp3 = uInput3
        date_obj = datetime.datetime.strptime(tmp3, date_format)
    except ValueError:
        print("Incorrect date format, should be YYYY-MM-DD")
    else:
        break
while(True):
    try:
        shortTerm = input("Enter your short-term moving average value: ")
        shortTerm = int(shortTerm)
        break
    except ValueError:
        print("You must enter an integer.")
    else:
        break

while(True):
    try:
        mediumTerm = input("Enter your mid-term moving average value: ")
        mediumTerm = int(mediumTerm)
        break
    except ValueError:
        print("You must enter an integer")
    else:
        break

while(True):
    try:
        longTerm = input("Enter your long-term moving average value: ")
        longTerm = int(longTerm)
        break
    except ValueError:
        print("You must enter an integer")
    else:
        break

filename = (tmp1 + '.csv')

# Download stock data then export as CSV
data_df = yf.download(tmp1, start=tmp2, end=tmp3)
data_df.to_csv(filename)

# Wait until file download completes
pwd  = os.getcwd()
path = (pwd + "/" + filename)
while not os.path.exists(path):
    time.sleep(1)
    print("%s not ready" % path)
else:
    print("%s complete" % path)

# Read in data
underlying = pd.read_csv(filename)

# Visualize the data
plt.figure(figsize=(13, 7))
plt.plot(underlying['Close'], label = tmp1)
plt.title(tmp1 + ' Closing Price History')
plt.xlabel(tmp2 + ' through ' + tmp3)
plt.ylabel('Closing Price USD ($)')
plt.legend(loc = 'upper left')
# plt.show()

# Create the simple short-term moving average
SMAshort = pd.DataFrame()
SMAshort['Close'] = underlying['Close'].rolling(window = shortTerm).mean()

# Create the simple mid-term moving average
SMAmid = pd.DataFrame()
SMAmid['Close'] = underlying['Close'].rolling(window = mediumTerm).mean()

# Create the simple long-term moving average (for the MACD)
SMAlong = pd.DataFrame()
SMAlong['Close'] = underlying['Close'].rolling(window=longTerm).mean()

# Visualize the data
plt.figure(figsize=(13, 7))
plt.plot(underlying['Close'], label = tmp1)
plt.plot(SMAshort['Close'], label = 'Short-term SMA')
plt.plot(SMAmid['Close'], label = 'Medium-term SMA')
plt.plot(SMAlong['Close'], label = 'Long-term SMA')
plt.title(tmp1 + ' Closing Price History')
plt.xlabel(tmp2 + ' through ' + tmp3)
plt.ylabel('Closing Price USD ($)')
plt.legend(loc = 'upper left')
#plt.show()

# Create a new dataframe to combine everything
data = pd.DataFrame()
data['underlying'] = underlying['Close']
data['SMAshort'] = SMAshort['Close']
data['SMAmid'] = SMAmid['Close']
data['SMAlong'] = SMAlong['Close']
# The MACD is the difference between the mid and long SMA values
data['MACD'] = (SMAmid['Close'] - SMAlong['Close'])

# Create a function to give 'arrows' when to buy or sell
def buy_sell(data):
    sigPriceSell=[]
    sigPriceBuy=[]
    flag = -1

    for i in range(len(data)):
        # As the shortSMA passes over the midSMA and is above the longSMA
        if data['SMAshort'][i] > data['SMAmid'][i] and data['SMAshort'][i] > data['SMAlong'][i]:
            if flag != 1:
                sigPriceBuy.append(data['underlying'][i])
                sigPriceSell.append(np.nan) #append nothing
                flag = 1 #I was here
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
                # As the shortSMA passes over the midSMA and is below the longSMA
        elif data['SMAshort'][i] < data['SMAmid'][i] and data['SMAshort'][i] < data['SMAlong'][i]:
            if flag != 0:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(data['underlying'][i])
                flag = 0
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)
    return(sigPriceBuy, sigPriceSell)

# Store the buy and sell data into a variable
buy_sell = buy_sell(data)
data['Buy_Signal_Price'] = buy_sell[0]
data['Sell_Signal_Price'] = buy_sell[1]

# Visualize the data and the strategy to buy or sell based on your SMA value inputs
plt.figure(figsize=(14, 8))
plt.title(str(shortTerm) + ", " + str(mediumTerm) + ", " + str(longTerm) + " SMA MACD for Ticker: " + tmp1)
plt.plot(data['underlying'], label = tmp1, alpha = 0.5)
plt.plot(data['SMAshort'], label = str(shortTerm) + " SMA", alpha = 0.5)
plt.plot(data['SMAmid'], label = str(mediumTerm) + " SMA", alpha = 0.5)
plt.plot(data['SMAlong'], label = str(longTerm) + " SMA", alpha = 0.5)
plt.plot(data['MACD'], label = "MACD")
plt.scatter(data.index, data['Buy_Signal_Price'], label = 'Buy', marker = '^', color = 'green')
plt.scatter(data.index, data['Sell_Signal_Price'], label = 'Sell', marker = 'v', color = 'red')
plt.xlabel('Total Days from ' + tmp2 + ' through ' + tmp3)
plt.ylabel('Closing Price USD ($)')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = len(data), colors = 'blue', linestyles='solid', label='MACD', data = None, alpha = 0.6)
plt.savefig((str(shortTerm) + "_" + str(mediumTerm) + "_" + str(longTerm) + "_SMA_MACD_for_Ticker_" + tmp1 + "_"+ tmp2 + '_through_' + tmp3 + ".png"), dpi=None, facecolor='w', edgecolor='w',orientation='portrait', format=None, transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)
#plt.show()

##################################################################################################
# LSTM

filename = (tmp1 + '.csv')

# Download stock data then export as CSV
data_df = yf.download(tmp1, start=tmp2, end=tmp3)
data_df.to_csv(filename)

# Wait until file download completes
pwd  = os.getcwd()
path = (pwd + "/" + filename)
while not os.path.exists(path):
    time.sleep(1)
    print("%s not ready" % path)
else:
    print("%s complete" % path)

# Read in data
underlying = pd.read_csv(filename)
print (underlying.head(5))

# Determine nature of data
size = underlying.size
shape = underlying.shape

#print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}".
#format(size, shape, shape[0]*shape[1])

