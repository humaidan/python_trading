import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import datetime
import copy
from pathlib import Path
import pickle
from TAfunctions import *
import matplotlib.pyplot as plt


filepath = 'output/s49_data.pkl'
days = 365*4

tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL1-USD', 'XRP-USD', 
           'DOT1-USD', 'LINK-USD', 'LTC-USD', 'ALGO-USD', 'VET-USD', 
           'MATIC-USD']

ohlc_mon = {} # directory with ohlc value for each stock

if Path(filepath).is_file():
  print('Reading saved data ...')
  with open(filepath, 'rb') as handle:
    ohlc_mon = pickle.load(handle)

else:
  print('Downloading fresh data ... please wait')
  attempt = 0 # initializing passthrough variable
  drop = [] # initializing list to store tickers whose close price was successfully extracted
  while len(tickers) != 0 and attempt <= 5:
      tickers = [j for j in tickers if j not in drop] # removing stocks whose data has been extracted from the ticker list
      for i in range(len(tickers)):
          try:
              ohlc_mon[tickers[i]] = pdr.get_data_yahoo(
                                  tickers[i],
                                  datetime.date.today()-datetime.timedelta(days),
                                  datetime.date.today(),
                                  interval='m')
              ohlc_mon[tickers[i]].dropna(inplace = True)
              drop.append(tickers[i])       
          except:
              print(tickers[i]," :failed to fetch data...retrying")
              continue
      attempt+=1

  with open(filepath, 'wb') as handle:
    pickle.dump(ohlc_mon, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

tickers = ohlc_mon.keys() # redefine tickers variable after removing any tickers with corrupted data

################################Backtesting####################################
# print(tickers)
# print(ohlc_mon)

ohlv_dic = copy.deepcopy(ohlc_mon)
return_df = pd.DataFrame()

for ticker in tickers:
  ohlv_dic[ticker]['mon_ret'] = ohlv_dic[ticker]['Adj Close'].pct_change()
  return_df[ticker] = ohlv_dic[ticker]['mon_ret']

#print(return_df)

def pflio(DF,m,x):
    """Returns cumulative portfolio return
    DF = dataframe with monthly return info for all stocks
    m = number of stock in the portfolio
    x = number of underperforming stocks to be removed from portfolio monthly"""
    df = DF.copy()
    portfolio = []
    monthly_ret = [0]
    for i in range(1,len(df)):
        if len(portfolio) > 0:
            monthly_ret.append(df[portfolio].iloc[i,:].mean())
            bad_stocks = df[portfolio].iloc[i,:].sort_values(ascending=True)[:x].index.values.tolist()
            portfolio = [t for t in portfolio if t not in bad_stocks]
        fill = m - len(portfolio)
        new_picks = df.iloc[i,:].sort_values(ascending=False)[:fill].index.values.tolist()
        portfolio = portfolio + new_picks
        #print(portfolio)
    monthly_ret_df = pd.DataFrame(np.array(monthly_ret),columns=["mon_ret"])
    return monthly_ret_df

myportfolio = pflio(return_df, 6, 3)

BTC = pdr.get_data_yahoo("BTC-USD",datetime.date.today()-datetime.timedelta(days),datetime.date.today(),interval='m')
BTC["mon_ret"] = BTC["Adj Close"].pct_change()

print('CAGR = ', CAGR(myportfolio))
print('Sharpe = ', sharpe(myportfolio, 0.025))
print('dd = ', max_dd(myportfolio))


#visualization
fig, ax = plt.subplots(sharex=True, sharey=False)
plt.plot((1+pflio(return_df,6,3)).cumprod())
plt.plot((1+BTC["mon_ret"][2:].reset_index(drop=True)).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumulative return")
plt.xlabel("months")
ax.legend(["Strategy Return","Index Return"])

plt.show()