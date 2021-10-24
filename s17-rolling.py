from os import close
import pandas as pd
from pandas.io.pytables import Term
from yahoofinancials import YahooFinancials
import datetime
import matplotlib.pyplot as plt

all_tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']

# extracting stock data (historical close price) for the stocks identified
close_prices = pd.DataFrame()
end_date = (datetime.date.today()).strftime('%Y-%m-%d')
beg_date = (datetime.date.today()-datetime.timedelta(365*4)).strftime('%Y-%m-%d')
cp_tickers = all_tickers
attempt = 0
drop = []
while len(cp_tickers) != 0 and attempt <=5:
    print("-----------------")
    print("attempt number ",attempt)
    print("-----------------")

    cp_tickers = [j for j in cp_tickers if j not in drop]

    for i in range(len(cp_tickers)):
        try:
            yahoo_financials = YahooFinancials(cp_tickers[i])
            json_obj = yahoo_financials.get_historical_price_data(beg_date,end_date,"daily")
            ohlv = json_obj[cp_tickers[i]]['prices']
            temp = pd.DataFrame(ohlv)[["formatted_date","adjclose"]]
            temp.set_index("formatted_date",inplace=True)
            temp2 = temp[~temp.index.duplicated(keep='first')]
            close_prices[cp_tickers[i]] = temp2["adjclose"]
            drop.append(cp_tickers[i])       
        except:
            print(cp_tickers[i]," :failed to fetch data...retrying")
            continue
    attempt+=1

#print(close_prices)

daily_returns = close_prices.pct_change()

#print(daily_returns)

#print(daily_returns.rolling(20).mean())

#print(daily_returns.ewm(span=20, min_periods=20).mean().tail(50))

cp_stadardized = ( close_prices - close_prices.mean() ) / close_prices.std()
cp_stadardized.plot(subplots=True, layout=(3,2), title="Top Cryptos Std Deviation", grid=True)

plt.show()