#Description: This program uses an LSTM model to predict weekly/monthly data based on last 60 period


#import the libraries
print('Importing libraries ...')
import requests
from pathlib import Path
import csv
from bs4 import BeautifulSoup
import math
from numpy.core.einsumfunc import einsum_path
import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings("ignore")

num_periods_to_check = 12
resample_freq = '5D'
cryptoBook = []

dirpath = 'output/ML2/' + datetime.date.today().strftime('%y%m%d') + '/'
Path(dirpath).mkdir(parents=True, exist_ok=True)

csv_date = datetime.date.today().strftime('%d%b%y')
csv_file = dirpath + 'cryptoBook-' + csv_date + '.csv'
csv_cols = ['Ticker', 'LastPrice', 'Prediction', 'rmse']



def getYahooTopCrptos(x=100):
  tickers = []
  x = x + 2   #account for USDT & USDSC

  url = 'https://finance.yahoo.com/cryptocurrencies/?offset=0&count=' + str(x)
  headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
  page = requests.get(url, headers=headers, timeout=5) 
  page_content = page.content
  
  soup = BeautifulSoup(page_content, 'html.parser')
  
  cells = soup.find_all('td', attrs={'aria-label': 'Symbol'})
  for cell in cells:
    ticker = cell.get_text()
    if ticker!='USDT-USD' and ticker!='USDC-USD':
      tickers.append(ticker)

  return tickers


# Download historical data for required stocks
#tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD']
tickers = getYahooTopCrptos(200)
#tickers = ['XRP-USD', 'ALGO-USD', 'NANO-USD', 'SFMS-USD', 'BTC-USD', 'ETH-USD']
#tickers = tickers[-4:]  #FOR TESTING PURPOSES

loop = 0

#for f in freq:
for ticker in tickers:

  print()
  print()
  #print('*****  [' + str(loop) + '] Processing ' + ticker + ' data ...   ******')
  print('*****  [' + str(loop) + '] Processing ' + 
            '\x1b[6;30;42m' + ticker + '\x1b[0m' +
            ' data ...   ******')
  loop = loop + 1

  #print('Downloading ' + ticker + ' data ...')
  df = pdr.data.get_data_yahoo(ticker,
              #datetime.date.today()-datetime.timedelta(365*7),
              '2010-01-01',
              datetime.date.today() )
              #interval='m')
  #print('[Done]')

  # endOfMonth = datetime.date.today() - relativedelta(months=+1)
  # endOfMonth = pd.Period(endOfMonth,freq='M').end_time.date()

  # df = df[ : endOfMonth ].resample('M').mean()
  df = df.resample(resample_freq).mean()
  #print(df)

  if len(df) < num_periods_to_check:
    print(' ..... skipping - available data: ' + str(len(df)))
    continue

  #Create new dataframe with only the Close column
  data = df.filter(['Close'])
  #print(data)
  print('***** Processing ' + str(df.shape[0]) + ' entries')

  #Convert the dataframe to numpy array
  dataset = data.values

  #######################   PREP THE MODEL DATA  #######################

  #Get the number of rows to train the model on
  training_data_len = math.ceil(len(dataset) * .8)

  #Scale the data
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  #Create the training data set

  #Create the scaled training data set
  train_data = scaled_data[0:training_data_len,:]

  #Split the data into x_train and y_train data sets
  x_train = []    #independent training variables
  y_train = []    #target variables

  for i in range(num_periods_to_check, len(train_data)):
    x_train.append(train_data[i-num_periods_to_check:i, 0])
    y_train.append(train_data[i, 0])

  #Convert x_train & y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  #Reshape the data to expectation of LSTM input of 3D array (#samples, #time-steps, features)
  # print('Training ' + str(x_train.shape[0]) + ' samples with ' + str(x_train.shape[1]) + ' timesteps and 1 feature .....')
  # print(x_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


  #######################   LSTM MODEL  #######################

  #Build the LSTM model
  model = Sequential()

  #LSTM model with 50 neurons + other layers
  model.add( LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)) )
  model.add( LSTM(50, return_sequences=False) )
  model.add( Dense(25) )
  model.add( Dense(1) )

  #Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  #Train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)


  #######################  TESTING STAGE  #######################
  #Create the testing data

  #Create a new array containing scaled values for the rest of 20% of data (index from training_data_len onwards till the end)
  #not scaled 
  test_data = scaled_data[training_data_len-num_periods_to_check:, :]
  x_test = []
  y_test = dataset[training_data_len:,:]

  for i in range (num_periods_to_check, len(test_data)):
    x_test.append(test_data[i-num_periods_to_check:i, 0])

  # Scale test data
  #Convert the data to a numpy array
  x_test = np.array(x_test)

  #Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  #Get the model's predicted price values (for the 20% data)
  predictions = model.predict(x_test)
  # un-scale data into the same as  y_data dataset contains
  predictions = scaler.inverse_transform(predictions)


  #Evaluate model - stat
  #Get the root mean squared error (RMSE)
  rmse = np.sqrt( np.mean( predictions - y_test)**2 )
  print('RMSE=', rmse)


  # Plot the data
  train = data[:training_data_len]
  valid = data[training_data_len:]
  valid['Predict'] = predictions

  #Visualize the data
  plt.figure(figsize=(16,8))
  plt.title(ticker[0:3] + ' LSTM Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price ($)', fontsize=18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close', 'Predict']])
  plt.legend(['Train', 'Actuals', 'Prediction'], loc='lower right')
  #plt.figtext(0.0,0.0, 'rmse=' + str(math.ceil(rmse)), fontsize=8, va="top", ha="left")

  plt.savefig(dirpath + ticker + '.png')
  #plt.show()



  ##### Before Prediction re-train on full dataset
  #training_data_len = math.ceil(len(dataset) )

  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(dataset)

  train_data = scaled_data #[0:training_data_len,:]
  
  x_train = []    #independent training variables
  y_train = []    #target variables

  for i in range(num_periods_to_check, len(train_data)):
    x_train.append(train_data[i-num_periods_to_check:i, 0])
    y_train.append(train_data[i, 0])
    
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

  #Build the LSTM model
  model = Sequential()
  
  model.add( LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)) )
  model.add( LSTM(50, return_sequences=False) )
  model.add( Dense(25) )
  model.add( Dense(1) )
  
  model.compile(optimizer='adam', loss='mean_squared_error')
  
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  ### Model is re-trained now

  ###### Predict next month closing price
  last_60_days = data[-num_periods_to_check:].values
  last_60_days_scaled = scaler.transform(last_60_days)

  x_test = []
  x_test.append(last_60_days_scaled)
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  pred_price = model.predict(x_test)
  pred_price = scaler.inverse_transform(pred_price)

  newday = data.index[-1] + datetime.timedelta(7)

  # endOfMonth = data.index[-1] + relativedelta(months=+1)
  # endOfMonth = pd.Period(endOfMonth,freq='M').end_time.date()
  # newday = endOfMonth

  # print('data:')
  # print(data)
  print(ticker + ': ' + str(newday) + '  --> $' + str(pred_price[0][0]))
  
  cryptoBook.append(
    {
      'Ticker' : ticker,
      'LastPrice' : float(last_60_days[-1]),
      'Prediction' : float(pred_price),
      'rmse' : rmse
      }
  )

print()
#print(cryptoBook)
#write results
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
        writer.writeheader()
        for data in cryptoBook:
            writer.writerow(data)
except IOError:
    print("I/O error")
