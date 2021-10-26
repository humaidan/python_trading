import pandas as pd
import pandas_datareader.data as pdr
import datetime
import numpy as np
import matplotlib.pyplot as plt



ticker = "XRP-USD"
ohlcv = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(365*2),datetime.date.today())

def GACR(DF):
    df = DF.copy()
    df['daily_ret'] = DF['Adj Close'].pct_change()
    df['cum_return'] = (1 + df['daily_ret']).cumprod()
    n = len(df)/252;
    GACR = (df['cum_return'][-1])**(1/n) - 1
    return GACR

def Volatility(DF):
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    vol = df['daily_ret'].std() * np.sqrt(252)
    return vol

df = ohlcv.copy()
df['GACR'] = GACR(df)
print(df['GACR'])

df['Vol'] = Volatility(df)
print(df['Vol'])


# df[['Adj Close', 'GACR', 'Vol']].plot(title='XRP Volatility', subplots=True, 
#             style=['black', 'blue', 'green'],
#             sharex=True, sharey=False, legend=False)
# for ax in plt.gcf().axes:
#     ax.legend(loc=1)

# plt.show()

