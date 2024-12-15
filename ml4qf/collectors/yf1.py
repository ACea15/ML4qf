import pandas as pd
import yfinance as yf

import requests
from bs4 import BeautifulSoup
import bs4 as bs


# Step 1: Scrape IBEX 35 tickers from a website
url = "https://en.wikipedia.org/wiki/IBEX_35"  # Wikipedia page for IBEX 35 components
url = "https://en.wikipedia.org/wiki/FTSE_100_Index"  # Wikipedia page for IBEX 35 components
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#url = "https://en.wikipedia.org/wiki/DAX"
# html = requests.get(url)
# soup = bs.BeautifulSoup(html.text, 'lxml')
# tables = soup.find_all('table', {'class': 'wikitable sortable'})
# for ti in tables[0]:
#     tickers = []
#     try:
#         for row in ti.findAll('tr')[0]:
#             ticker = row.findAll('td').text
#             ticker = ticker[:-1]
#             tickers.append(ticker)
#     except IndexError:
#         continue

#url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the components
#table = soup.find('table', {'class': 'wikitable'})
tables = soup.find_all('table', {'class': 'wikitable'})
table = tables[0]
tickers = []
sectors = []
for row in table.find_all('tr')[1:]:
    # print(row.find_all('td')[2].text)
    # ticker = row.find_all('td')[1].text.strip().split()[0] #.replace(".MC", "") + ".MC"  # Yahoo Finance ticker format
    # tickers.append(ticker)
    cells = row.find_all('td')
    if len(cells) > 1:
        ticker = cells[2].text.strip() #+ '.MC'  # Adjusting to match Yahoo Finance's requirements
        tickers.append(ticker)

print("IBEX 35 Tickers:")
print(tickers)



class Inputs(dict):
    """Represents configuration options, works like a dict."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            self[name] = Inputs({})
            return self[name]

    def __setattr__(self, name, val):
        self[name] = val



INDEX = Inputs()
INDEX.ibex35.url = "https://en.wikipedia.org/wiki/IBEX_35"
INDEX.ibex35.num_stocks = 35
INDEX.ibex35.ticker_index = None
INDEX.ibex35.company_index = None
INDEX.ibex35.sector_index = None
###
INDEX.sp500.url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
INDEX.sp500.num_stocks = 500
INDEX.sp500.ticker_index = 0
INDEX.sp500.company_index = 1
INDEX.sp500.sector_index = 2
###
INDEX.ftse100.url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
INDEX.ftse100.num_stocks = 100
INDEX.ftse100.ticker_index = 1
INDEX.ftse100.company_index = 0
INDEX.ftse100.sector_index = 2
###
INDEX.dax40.url = "https://en.wikipedia.org/wiki/DAX"
INDEX.dax40.num_stocks = 40
INDEX.dax40.ticker_index = 3
INDEX.dax40.company_index = 1
INDEX.dax40.sector_index = 2


# Find the table containing the components
#table = soup.find('table', {'class': 'wikitable'})

def readwiki_index(index: Inputs):
    
    response = requests.get(index.url)
    soup = BeautifulSoup(response.text, 'html.parser')    
    tables = soup.find_all('table', {'class': 'wikitable'})
    for table in tables:
        try:
            tickers = []
            sectors = []
            companies = []
            tr = table.find_all('tr')
            tr0 = tr[0].text.split()
            if index.ticker_index is None:
                ticker_index = [i for i, _ in enumerate(tr0) if "ticker" in _.lower()][0]
            else:
                ticker_index = index.ticker_index
            if index.sector_index is None:
                sector_index = [i for i, _ in enumerate(tr0) if "sector" in _.lower()][0]
            else:
                sector_index = index.sector_index
            if index.company_index is None:
                company_index = [i for i, _ in enumerate(tr0) if ("company" in _.lower() or "security" in _.lower())][0]
            else:
                company_index = index.company_index
            for row in tr[1:]:
                # print(row.find_all('td')[2].text)
                # ticker = row.find_all('td')[0].text.strip().split()[0] #.replace(".MC", "") + ".MC"  # Yahoo Finance ticker format
                # tickers.append(ticker)
                cells = row.find_all('td')
                if len(cells) > 1:
                    ticker = cells[ticker_index].text.strip() #+ '.MC'  # Adjusting to match Yahoo Finance's requirements
                    tickers.append(ticker)
                    company = cells[company_index].text.strip() #+ '.MC'  # Adjusting to match Yahoo Finance's requirements
                    companies.append(company)
                    sector = cells[sector_index].text.strip() #+ '.MC'  # Adjusting to match Yahoo Finance's requirements
                    sectors.append(sector)

            if abs(len(tickers) - index.num_stocks) / index.num_stocks < 0.01:  # success
                break
        except (IndexError, AttributeError):
            continue

    return tickers, companies, sectors

def build_indexes(INDEX: Inputs):

    info_index = Inputs()
    for k, v in INDEX.items():
        tickers, companies, sectors = readwiki_index(v)
        info_index[k] = Inputs(tickers=tickers, companies=companies,sectors=sectors)
    return info_index


index_info = build_indexes(INDEX)

tickers, companies, sectors = readwiki_index(INDEX.ibex35)
print(tickers)
print(companies)
print(sectors)




if __name__ == "__main__":
    ticker = "AAPL"  # Apple Inc.
    start_date = "2022-01-01"
    end_date = "2023-01-01"

    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Display the first few rows of the historical data
    print(data.head())

    
    # Initialize a Ticker object for Apple Inc.
    ticker = yf.Ticker("AAPL")

    # Fetch historical data
    historical_data = ticker.history(period="1y")
    print("Historical Data:")
    print(historical_data.head())

    # Fetch dividends
    dividends = ticker.dividends
    print("\nDividends:")
    print(dividends)

    # Fetch stock splits
    stock_splits = ticker.splits
    print("\nStock Splits:")
    print(stock_splits)

    # Fetch financial statements
    print("\nIncome Statement:")
    print(ticker.financials)

    print("\nBalance Sheet:")
    print(ticker.balance_sheet)

    print("\nCash Flow Statement:")
    print(ticker.cashflow)

    # Fetch earnings data
    print("\nEarnings:")
    print(ticker.earnings)

    # print("\nQuartely Earnings:")
    # print(ticker.quarterly_earnings)
    print("\nQuartely Earnings:")
    print(ticker.income_stmt)


    # Fetch analyst recommendations
    print("\nAnalyst Recommendations:")
    print(ticker.recommendations)

    # Fetch options data
    expiration_dates = ticker.options
    print("\nOptions Expiration Dates:")
    print(expiration_dates)

    # Fetch option chain for the first expiration date
    option_chain = ticker.option_chain(expiration_dates[0])
    print("\nOptions Chain for the first expiration date:")
    print(option_chain.calls.head())
    print(option_chain.puts.head())

    # Fetch basic stock info
    print("\nBasic Info:")
    print(ticker.info)
