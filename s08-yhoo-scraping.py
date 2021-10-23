from yahoofinancials import YahooFinancials
import json

ticker = 'BTC-USD'
yahoo_financials = YahooFinancials(ticker)

# balance_sheet_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'balance')
# income_statement_data_qt = yahoo_financials.get_financial_stmts('quarterly', 'income')
# all_statement_data_qt =  yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])
# apple_earnings_data = yahoo_financials.get_stock_earnings_data()
# apple_net_income = yahoo_financials.get_net_income()

historical_stock_prices = yahoo_financials.get_historical_price_data('2018-10-01', '2020-10-23', 'monthly')

print(historical_stock_prices)

# print('type: {}'.format(type(historical_stock_prices)))

# x = len(historical_stock_prices['BTC-USD']['prices'])
# print('Length of prices: {}'.format(str(x)))

with open('output/08_btc.json', 'w') as outfile:
    json.dump(historical_stock_prices, outfile)
