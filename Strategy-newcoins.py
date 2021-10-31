 #This example uses Python 2.7 and the python-request library.

import datetime
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from Keys import coin_marketcap_api_key
from datetime import date, timedelta
from pathlib import Path
import pickle

filepath = 'output/Strategy-newcoins.pkl'

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coin_marketcap_api_key,
}

session = Session()
session.headers.update(headers)

data = {}

if Path(filepath).is_file():
  print('Reading saved data ...')
  with open(filepath, 'rb') as handle:
    data = pickle.load(handle)

else:
  try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)

    today = date.today()
    yesterday = str(today - timedelta(days=1))
    yesterday_datetime = datetime.datetime.strptime(yesterday, '%Y-%m-%d')

    #two keys [data & status]
    for entry in data['data']:
      symbol = entry['symbol']
      date_added_str = entry['date_added'][:10]

      date_added = datetime.datetime.strptime(date_added_str, '%Y-%m-%d')

      if yesterday_datetime < date_added:
        print(symbol + ': ' + date_added_str)

    


    with open(filepath, 'wb') as handle:
      pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print(data)

  except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)