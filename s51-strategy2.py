import numpy as np
import pandas as pd
import copy
import pickle
import datetime
import matplotlib.pyplot as plt
import cryptocompare


def ATR(DF,n):
    'function to calculate True Range and Average True Range'
    df = DF.copy()
    df['H-L']=abs(df['high']-df['low'])
    df['H-PC']=abs(df['high']-df['close'].shift(1))
    df['L-PC']=abs(df['low']-df['close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2['ATR']

def CAGR(DF):
    'function to calculate the Cumulative Annual Growth Rate of a trading strategy'
    df = DF.copy()
    df['cum_return'] = (1 + df['ret']).cumprod()
    n = len(df)/(365*24)
    CAGR = (df['cum_return'].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    'function to calculate annualized volatility of a trading strategy'
    df = DF.copy()
    vol = df['ret'].std() * np.sqrt(365*24)
    return vol

def sharpe(DF,rf):
    'function to calculate sharpe ratio ; rf is the risk free rate'
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    'function to calculate max drawdown'
    df = DF.copy()
    df['cum_return'] = (1 + df['ret']).cumprod()
    df['cum_roll_max'] = df['cum_return'].cummax()
    df['drawdown'] = df['cum_roll_max'] - df['cum_return']
    df['drawdown_pct'] = df['drawdown']/df['cum_roll_max']
    max_dd = df['drawdown_pct'].max()
    return max_dd

# Download historical data (monthly) for selected stocks

cryptocompare.cryptocompare._set_api_key_parameter('fae29551d5d653e9927fae7b4bd527153edb659f9e87255ec072e6c993c06b17')
# df = pd.DataFrame(cryptocompare.get_historical_price_hour('XRP', currency='USD', limit=24*7*4), columns=['time', 'close'])
# df['date'] = pd.to_datetime(df['time'], unit='s')
# df = df.set_index('date')

# df['close'].plot()
# plt.tight_layout()
# plt.title('Hourly XRP Price')
# plt.grid()

# print(df)
# print(df.dtypes)
# plt.show()



tickers = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL']

ohlc_intraday = {} # directory with ohlc value for each stock            
#key_path = 'D:\\Udemy\\Quantitative Investing Using Python\\1_Getting Data\\AlphaVantage\\key.txt'
#ts = TimeSeries(key=open(key_path,'r').read(), output_format='pandas')

attempt = 0 # initializing passthrough variable
drop = [] # initializing list to store tickers whose close price was successfully extracted
while len(tickers) != 0 and attempt <=5:
    tickers = [j for j in tickers if j not in drop]
    for i in range(len(tickers)):
        try:
            #ohlc_intraday[tickers[i]] = ts.get_intraday(symbol=tickers[i],interval='5min', outputsize='full')[0]
            ohlc_intraday[tickers[i]] = pd.DataFrame(
                        cryptocompare.get_historical_price_hour(
                            tickers[i], currency='USD', limit=24*7*4),
                            columns=['time', 'open','high','low','close','volumeto'])
            ohlc_intraday[tickers[i]]['date'] = pd.to_datetime(ohlc_intraday[tickers[i]]['time'], unit='s')
            ohlc_intraday[tickers[i]]['time'] = ohlc_intraday[tickers[i]]['date']
            ohlc_intraday[tickers[i]] = ohlc_intraday[tickers[i]].set_index('date')
            #ohlc_intraday[tickers[i]].drop('time', axis=1, inplace=True)
            ohlc_intraday[tickers[i]] = ohlc_intraday[tickers[i]].rename({'volumeto': 'volume'}, axis=1)  # new method
            # ohlc_intraday[tickers[i]].columns = ['Open','High','Low','Adj Close','Volume']
            #ohlc_intraday[tickers[i]].columns = ['open','high','low','close','volumeto']
            drop.append(tickers[i])
        except:
            print(tickers[i],' :failed to fetch data...retrying')
            continue
    attempt+=1

 
tickers = ohlc_intraday.keys() # redefine tickers variable after removing any tickers with corrupted data

################################Backtesting####################################

# calculating ATR and rolling max price for each stock and consolidating this info by stock in a separate dataframe
ohlc_dict = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}
for ticker in tickers:
    print('calculating ATR and rolling max price for ',ticker)
    ohlc_dict[ticker]['ATR'] = ATR(ohlc_dict[ticker],20)
    ohlc_dict[ticker]['roll_max_cp'] = ohlc_dict[ticker]['high'].rolling(20).max()
    ohlc_dict[ticker]['roll_min_cp'] = ohlc_dict[ticker]['low'].rolling(20).min()
    ohlc_dict[ticker]['roll_max_vol'] = ohlc_dict[ticker]['volume'].rolling(20).max()
    ohlc_dict[ticker].dropna(inplace=True)
    tickers_signal[ticker] = ''
    tickers_ret[ticker] = []


# identifying signals and calculating daily return (stop loss factored in)
for ticker in tickers:
    print('calculating returns for ',ticker)
    for i in range(len(ohlc_dict[ticker])):
        if tickers_signal[ticker] == '':
            tickers_ret[ticker].append(0)
            if ohlc_dict[ticker]['high'][i]>=ohlc_dict[ticker]['roll_max_cp'][i] and \
               ohlc_dict[ticker]['volume'][i]>1.5*ohlc_dict[ticker]['roll_max_vol'][i-1]:
                tickers_signal[ticker] = 'Buy'
                print('BUY --> ', ticker, ' : ', ohlc_dict[ticker]['time'][i], ' at ', ohlc_dict[ticker]['high'][i])
            elif ohlc_dict[ticker]['low'][i]<=ohlc_dict[ticker]['roll_min_cp'][i] and \
               ohlc_dict[ticker]['volume'][i]>1.5*ohlc_dict[ticker]['roll_max_vol'][i-1]:
                tickers_signal[ticker] = 'Sell'
                print('SELL --> ', ticker, ' : ', ohlc_dict[ticker]['time'][i], ' at ', ohlc_dict[ticker]['high'][i])
        
        elif tickers_signal[ticker] == 'Buy':
            if ohlc_dict[ticker]['close'][i]<ohlc_dict[ticker]['close'][i-1] - ohlc_dict[ticker]['ATR'][i-1]:
                tickers_signal[ticker] = ''
                tickers_ret[ticker].append(((ohlc_dict[ticker]['close'][i-1] - ohlc_dict[ticker]['ATR'][i-1])/ohlc_dict[ticker]['close'][i-1])-1)
            elif ohlc_dict[ticker]['low'][i]<=ohlc_dict[ticker]['roll_min_cp'][i] and \
               ohlc_dict[ticker]['volume'][i]>1.5*ohlc_dict[ticker]['roll_max_vol'][i-1]:
                tickers_signal[ticker] = 'Sell'
                tickers_ret[ticker].append(((ohlc_dict[ticker]['close'][i-1] - ohlc_dict[ticker]['ATR'][i-1])/ohlc_dict[ticker]['close'][i-1])-1)
            else:
                tickers_ret[ticker].append((ohlc_dict[ticker]['close'][i]/ohlc_dict[ticker]['close'][i-1])-1)
                
        elif tickers_signal[ticker] == 'Sell':
            if ohlc_dict[ticker]['close'][i]>ohlc_dict[ticker]['close'][i-1] + ohlc_dict[ticker]['ATR'][i-1]:
                tickers_signal[ticker] = ''
                tickers_ret[ticker].append((ohlc_dict[ticker]['close'][i-1]/(ohlc_dict[ticker]['close'][i-1] + ohlc_dict[ticker]['ATR'][i-1]))-1)
            elif ohlc_dict[ticker]['high'][i]>=ohlc_dict[ticker]['roll_max_cp'][i] and \
               ohlc_dict[ticker]['volume'][i]>1.5*ohlc_dict[ticker]['roll_max_vol'][i-1]:
                tickers_signal[ticker] = 'Buy'
                tickers_ret[ticker].append((ohlc_dict[ticker]['close'][i-1]/(ohlc_dict[ticker]['close'][i-1] + ohlc_dict[ticker]['ATR'][i-1]))-1)
            else:
                tickers_ret[ticker].append((ohlc_dict[ticker]['close'][i-1]/ohlc_dict[ticker]['close'][i])-1)
                
    ohlc_dict[ticker]['ret'] = np.array(tickers_ret[ticker])


# calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_dict[ticker]['ret']
strategy_df['ret'] = strategy_df.mean(axis=1)
CAGR(strategy_df)
sharpe(strategy_df,0.025)
max_dd(strategy_df)  


# vizualization of strategy return
(1+strategy_df['ret']).cumprod().plot()


#calculating individual stock's KPIs
cagr = {}
sharpe_ratios = {}
max_drawdown = {}
for ticker in tickers:
    print('calculating KPIs for ',ticker)      
    cagr[ticker] =  CAGR(ohlc_dict[ticker])
    sharpe_ratios[ticker] =  sharpe(ohlc_dict[ticker],0.025)
    max_drawdown[ticker] =  max_dd(ohlc_dict[ticker])

KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=['Return','Sharpe Ratio','Max Drawdown'])      
KPI_df.T



plt.show()