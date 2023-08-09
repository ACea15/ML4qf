__all__= ["scrap_tickers_index",
          "get_tickers_info",
          "select_assets"] 
import pandas as pd
import yfinance as yf
import random
import bs4 as bs
import requests
import pathlib

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
                     info: list[str],
                     data_folder: pathlib.Path | str=None,
                     name_family='default'
                     ):

    tickers_info = {k: [] for k in info}
    data_file = None
    if data_folder is not None:
        folder_path = pathlib.Path(data_folder)
        if not folder_path.is_dir():
            print("***** Creating data Info folder *****")
            folder_path.mkdir(parents=True, exist_ok=True)
        data_file = folder_path / name_family
        if data_file.is_file():
            print("***** Loading data info from csv file *****")
            df = pd.read_csv(data_file, index_col=0)
            return df
    for i_ticker in tickers:
        try:
            ticker_object = yf.Ticker(i_ticker)
            #time.sleep(0.5)
            for i_info in info:
                try:
                    tickers_info[i_info].append(ticker_object.info[i_info])
                except:
                    tickers_info[i_info].append(float("nan"))
                    print(f"{i_ticker} has no {i_info}")
        except:
            print(f"{i_ticker} cannot be fetched by yahoo")
    df = pd.DataFrame(index=tickers, data=tickers_info)
    if data_file is not None:
        print("***** Saving info data to csv file *****")
        df.to_csv(data_file)
    return df

def select_assets(df_sorted,
                  asset_percentages: dict[str:tuple],
                  asset_names: list[str]=None):

    if asset_names is not None:
        df_out = df_sorted.loc[asset_names].sort_values('marketCap', ascending=False)
    else:
        num_total_assets = len(df_sorted)
        sectors = []
        bucket_total = 0
        indexes = list()
        bucket = dict()
        for k, v in asset_percentages.items():
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

