import pandas_datareader.data as pdr
import datetime as dt
import matplotlib.pyplot as plt

ticker = 'XRP-USD'
start_date = dt.date.today() - dt.timedelta(365)
end_date = dt.date.today()

data = pdr.get_data_yahoo(ticker, start_date, end_date, interval='w')

print(data)

data.to_csv('output/06_1ticker.csv')

df = data['Adj Close']
df.plot()
plt.title('XRP')
plt.ylabel('Price ($)')
plt.show()