import pandas as pd
import pandas_datareader.data as pdr
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Download historical data for required stocks
ticker = "XRP-USD"
ohlcv = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(30*8),datetime.date.today())


def OBV(DF):
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

df = pd.DataFrame()
df['Price'] = ohlcv['Adj Close']
df['OBV'] = OBV(ohlcv)

print(df['OBV'])

df.plot(title='XRP', subplots=True, style=['black', 'blue'],
            sharex=True, sharey=False, legend=False)
for ax in plt.gcf().axes:
    ax.legend(loc=1)

plt.tight_layout()

#plt.ylim(0, 100)
#plt.axhline(y=25, color='red', linestyle='--')
#plt.axhline(y=50, color='red', linestyle='--')
#plt.axhline(y=75, color='red', linestyle='--')
plt.show()

#plt.savefig("output/29_adx.png")
