import math
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas_datareader as pdr
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Download historical data for required stocks
#ticker = 'XRP-USD'
ticker = 'BTC-USD'

df = pdr.data.get_data_yahoo(ticker,
            datetime.date.today()-datetime.timedelta(365*7),
            datetime.date.today())

#print(df)
# print(df.shape)

# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * .8 )

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#create training dataset
#create scaled training dataset
train_data = scaled_data[0:training_data_len, :]
#split data into x_train & y_train
x_train = []
y_train = []
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

#convert x_train & y_train into numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Re-shape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
print('Training the model ...')
model.fit(x_train, y_train, batch_size=1, epochs=1)
print('    Done')

#Create teh testing data set
#  new array containing scaled values
test_data = scaled_data[training_data_len - 60 : , :]

# create datasets x_test & y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models the predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean( predictions - y_test)**2 )
print('rmse error = ', rmse)

#Plot the data 
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actuals', 'Predictions'], loc='lower right')

plt.show()

#Show the valid and predicted prices
#print(valid)

#predict next 60 days

new_df = df.filter('Close')


for i in range(1, 60):
  last_60_days = new_df[-60:].values
  scaler.clip = False 
  last_60_days_scaled = scaler.tranform(last_60_days)
  X_test = []
  X_test.append(last_60_days_scaled)
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  pred_price = model.predict(X_test)
  pred_price = scaler.inverse_transform(pred_price)

  new_df = new_df.append(pred_price, ignore_index = True)

print(new_df.tail(61))