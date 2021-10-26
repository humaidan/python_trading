import numpy as np

def CAGR_daily(DF):
    df = DF.copy()
    df['daily_ret'] = DF['Adj Close'].pct_change()
    df['cum_return'] = (1 + df['daily_ret']).cumprod()
    n = len(df)/252;
    GACR = (df['cum_return'][-1])**(1/n) - 1
    return GACR

def CAGR_monthly(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    n = len(df)/12
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def CAGR(DF, d=12):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    if d == 252:
      return CAGR_daily(DF)
    else:
      return CAGR_monthly(DF)

def volatility(DF, d=12):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["mon_ret"].std() * np.sqrt(d)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["mon_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd