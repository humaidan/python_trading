import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt



ticker = "XRP-USD"
ohlcv = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(30*8),datetime.date.today())

def slope(DF,n):
    df = DF['Adj Close'].copy()
    slopes = [i*0 for i in range(n-1)]

    for i in range(n,len(df)+1):
        y = df[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

df = pd.DataFrame()
df['Price'] = ohlcv['Adj Close']
df['slope'] = slope(ohlcv, 5)

df.plot()

plt.show()