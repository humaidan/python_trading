import pandas_datareader.data as pdr
import datetime
import matplotlib.pyplot as plt

# Download historical data for required stocks
ticker = "XRP-USD"
ohlcv = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(365),datetime.date.today())

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def BollBnd(DF,n):
  df = DF.copy()
  df['MA'] = df['Adj Close'].rolling(n).mean()
  df['BB_up'] = df['MA'] + 2*df['MA'].rolling(n).std()
  df['BB_dn'] = df['MA'] - 2*df['MA'].rolling(n).std()
  df['BB_width'] = df['BB_up'] - df['BB_dn']
  return df


df = ATR(ohlcv, 20)
df[["Adj Close", "TR", "ATR"]].plot()


df2 = BollBnd(ohlcv, 20)
print(df2)
#df2[["Adj Close", "MA", "BB_up", "BB_dn"]].plot()
df2.iloc[:,[-5, -4,-3,-2]].plot()
plt.show()
