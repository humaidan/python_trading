import requests
from bs4 import BeautifulSoup

ticker = 'AAPL'
url = 'https://finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
page = requests.get(url, headers=headers, timeout=5) 
page_content = page.content

soup = BeautifulSoup(page_content, 'html.parser')

tabl = soup.find_all("div", {"class" : "W(100%) Whs(nw)"})

print(tabl)
print(type(tabl))




