import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

from sklearn.preprocessing import MinMaxScaler

ticker = 'BTC-USD'

print('Downloading ' + ticker + ' data ...')
df = pdr.data.get_data_yahoo(ticker,
            datetime.date.today()-datetime.timedelta(365),
            datetime.date.today())

#print(df.describe())

Hi = np.array([df.iloc[:,0]])
Low = np.array([df.iloc[:,1]])
Close = np.array([df.iloc[:,3]])

plt.figure(1)
H, = plt.plot(Hi[0,:])
L, = plt.plot(Low[0,:])
C, = plt.plot(Close[0,:])

plt.legend([H,L,C], ['High', 'Low', 'Close'])
plt.show(block=False)

X = np.concatenate([Hi,Low], axis=0)

X = np.transpose(X)

Y = Close
Y = np.transpose(Y)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

scaler1 = MinMaxScaler()
scaler1.fit(Y)
Y = scaler1.transform(Y)

X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
#print(X.shape)

model = Sequential()
model.add(LSTM(100, activation='tanh', input_shape=(1,2), recurrent_activation='hard_sigmoid'))
model.add( Dense(1) )

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[metrics.mae])

#model.fit(X, Y, epochs=100, batch_size=1, verbose=2)
model.fit(X, Y, epochs=15, batch_size=1, verbose=2)

Predict=model.predict(X, verbose=1)
#print(Predict)

plt.figure(2)
plt.scatter(Y, Predict)
plt.show(block=False)

plt.figure(3)
Test, = plt.plot(Y)
Predict, = plt.plot(Predict)
plt.legend([Predict, Test], ['Predicted Data', 'Real Data'])
plt.show()