#Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM)
#               to predict the closing prices of crypto (BTC) using the past 60 days price

#import the libraries
import sys
import math
from numpy.core.einsumfunc import einsum_path
import pandas_datareader as pdr
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# Download historical data for required stocks
#ticker = 'XRP-USD'
ticker = 'BTC-USD'

print('Downloading ' + ticker + ' data ...')
df = pdr.data.get_data_yahoo(ticker,
            datetime.date.today()-datetime.timedelta(365*7),
            datetime.date.today())

# print(df)
# print(df.shape)

# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()


#Create new dataframe with only the Close column
data = df.filter(['Close'])

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

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

  # if i==61:
  #   print('looping i=', i)
  #   print(x_train)
  #   print(y_train)
  #   print()


#Convert x_train & y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data to expectation of LSTM input of 3D array (#samples, #time-steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#print(x_train)


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
#model.fit(x_train, y_train, batch_size=1, epochs=1)
model.fit(x_train, y_train, batch_size=32, epochs=2)



#######################  TESTING STAGE  #######################
#Create the testing data

#Create a new array containing scaled values for the rest of 20% of data (index from training_data_len onwards till the end)
#not scaled 
test_data = scaled_data[training_data_len-60:, :]
x_test = []
y_test = dataset[training_data_len:,:]

for i in range (60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

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
#print('RMSE=', rmse)


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
plt.figtext(0.0,0.0, 'rmse=' + str(math.ceil(rmse)), fontsize=8, va="top", ha="left")

plt.savefig('output/ML2-'+ticker+'-1.png')
#plt.show()


###### Predict tomorrows' closing price

# last_60_days = data[-60:].values
# last_60_days_scaled = scaler.transform(last_60_days)

# x_test = []
# x_test.append(last_60_days_scaled)
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# pred_price = model.predict(x_test)
# pred_price = scaler.inverse_transform(pred_price)

# print('Tomorrows Price: $' + str(pred_price))


##### Predict next 100 days
new_df = data


# print('start point..')
# print(new_df.tail(10))
# print('Press any key ...')
# sys.stdin.read(1)

for i in range(0, 100):
  
  newrow = new_df.iloc[-1]
  newday = new_df.index[-1] + datetime.timedelta(1)
  newrow.name = newday
  new_df = new_df.append(newrow)

  last_60_days = new_df[-60:].values
  last_60_days_scaled = scaler.transform(last_60_days)

  x_test = []
  x_test.append(last_60_days_scaled)
  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  pred_price = model.predict(x_test)
  pred_price = scaler.inverse_transform(pred_price)

  new_df.iloc[-1,-1] = pred_price[0][0]

  # print()
  # print('i=', i)
  # print(new_df.tail(10))
  # print('Press any key ...')
  # sys.stdin.read(1)
print(new_df[-100:])

##### Plot the data
current = data
future = new_df.iloc[-100]

new_df.to_csv('output/Strategy-ML2.csv')

#Visualize the data
plt.figure(figsize=(16,8))
plt.title(ticker[0:3] + ' LSTM Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($)', fontsize=18)
plt.plot(current['Close'])
plt.plot(future['Close'])
plt.legend(['Actuals', 'Prediction'], loc='lower right')
plt.figtext(0.0,0.0, 'rmse=' + str(math.ceil(rmse)), fontsize=8, va="top", ha="left")

plt.savefig('output/ML2-'+ticker+'-2.png')
plt.show()

