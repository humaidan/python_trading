import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


ticker = 'BTC-USD'

print('Downloading ' + ticker + ' data ...')
df = pdr.data.get_data_yahoo(ticker,
            datetime.date.today()-datetime.timedelta(365*7),
            datetime.date.today())
L = len(df)

Y = np.array([df.iloc[:,3]])

X1 = Y[:,0:L-5]
X2 = Y[:,1:L-4]
X3 = Y[:,2:L-3]


X = np.concatenate([X1,X2,X3], axis=0)
X = np.transpose(X)

#print(X)

Y = np.transpose(Y[:,3:L-2])

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler1 = MinMaxScaler()
scaler1.fit(Y)
Y = scaler1.transform(Y)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

model = Sequential()
model.add(LSTM(10, activation='tanh', input_shape=(1,3), recurrent_activation='hard_sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.mae])
model.fit(X_train, Y_train, epochs=15)

Predict = model.predict(X_test)

rmse = np.sqrt( np.mean( Predict - Y_test)**2 )
print('RMSE=', rmse)

print(Y_test)

plt.figure(2)
plt.scatter(Y_test, Predict)
plt.show(block=False)

plt.figure(3)
Test = plt.plot(Y_test)
Predict = plt.plot(Predict)
plt.show()