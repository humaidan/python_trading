# =============================================================================
# Import OHLCV data and calculate ATR and Supertrend technical indicators
# Author : Mayank Rasu

# Please report bug/issues in the Q&A section
# =============================================================================

# Import necesary libraries
import pandas_datareader.data as pdr
import numpy as np
import datetime

# Download historical data for required stocks
ticker = "AAPL"
ohlcv = pdr.get_data_yahoo(ticker,datetime.date.today()-datetime.timedelta(364),datetime.date.today())

    
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
    return df2.loc[:,['TR','ATR']]

def STREND(DF,m,n):
    """function to calculate Supertrend given historical candle data
        m = multiplier
        n = n day ATR"""
    df = DF.copy()
    df['ATR'] = ATR(df,n)['ATR'] # Usually ATR_5 is used for the calculation
    df["B-U"]=((df['High']+df['Low'])/2) + m*df['ATR'] 
    df["B-L"]=((df['High']+df['Low'])/2) - m*df['ATR']
    df["temp1"] = df["B-U"]
    df["temp2"] = df["B-L"]
    df["F-U"]= np.where((df["B-U"]<df["temp1"].shift(1))|(df["Adj Close"].shift(1)>df["temp1"].shift(1)),df["B-U"],df["temp1"].shift(1))
    df["F-L"]= np.where((df["B-L"]>df["temp2"].shift(1))|(df["Adj Close"].shift(1)<df["temp2"].shift(1)),df["B-L"],df["temp2"].shift(1))
    df["Strend"] = np.where(df["Adj Close"]<=df["F-U"],df["F-U"],df["F-L"])
    return df['Strend']