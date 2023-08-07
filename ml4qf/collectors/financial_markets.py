__all__= ["scrap_tickers_index",
          "get_tickers_info",
          "select_assets"] 
import pandas as pd
import yfinance as yf
import random
import bs4 as bs
import requests

def scrap_tickers_index(index_weblist: str) -> list[str]:
    html = requests.get(index_weblist)
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)
    return tickers

def get_tickers_info(tickers: list[str],
                     info: list[str]):

    tickers_info = {k: [] for k in info}
    for i_ticker in tickers:
        try:
            ticker_object = yf.Ticker(i_ticker)
            #time.sleep(0.5)
            for i_info in info:
                try:
                    tickers_info[i_info].append(ticker_object.info[i_info])
                except:
                    passed = False
                    tickers_info[i_info].append(float("nan"))
                    print(f"{i_ticker} has no {i_info}")
        except:
            print(f"{i_ticker} cannot be fetched by yahoo")
        
    return pd.DataFrame(index=tickers, data=tickers_info)

def select_assets(df_sorted, percentages: dict[str:tuple]):

    num_total_assets = len(df_sorted)
    sectors = []
    bucket_total = 0
    indexes = list()
    bucket = dict()
    for k, v in percentages.items():
        bucket_k = int(num_total_assets * v[0])
        bucket[k] = bucket_k
        assets_bucket = 0
        while assets_bucket < v[1]:
            index = random.randint(bucket_total,
                                   bucket_total + bucket_k)
            if (seci := df_sorted.iloc[index].sector) not in sectors:
                indexes.append(index)
                assets_bucket += 1
                sectors.append(seci)
        bucket_total += bucket_k
    df_out = df_sorted.iloc[indexes].sort_values('marketCap', ascending=False)
    return df_out
