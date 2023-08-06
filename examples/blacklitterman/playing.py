import pandas as pd
import yfinance as yf
import random

import statsmodels.api as sm
import getFamaFrenchFactors as gff
from datetime import date
import  numpy as np
from functools import partial


import bs4 as bs
import pickle
import requests

def get_factors(factor_names: list[str], frequency: str):
    df = None
    for fi in factor_names:
        obj_fi = getattr(gff, fi)
        df_fi = obj_fi(frequency=frequency)
        df_fi.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
        df_fi.set_index('Date', inplace=True)
        if df is None:
            df = df_fi
        else:
            df = df.join(df_fi)
    return df

def trim_df_date(df, start_date=None, end_date=None):
    if start_date is not None:
        start = np.where(df.index == start_date)[0][0]
    else:
        start = None
    if end_date is not None:
        end = np.where(df.index == end_date)[0][0]
    else:
        end = None
    df = df.iloc[start:end]
    return df

def regression_OLS(X, y):
    model = sm.OLS(y, X)
    model.fit()
    return model

def get_factor_names(factors):
    
    factor_names = []
    for k, v in factors.items():
        if isinstance(v, str):
            factor_names.append(v)
        elif isinstance(v, list):
            factor_names += v
    
def factors_regression(factor_names, df_gff, df_assets, regression_kernel):

    df_factors = df_gff[factor_names]
    X = sm.add_constant(df_factors)
    models = dict()
    for k, rets in df_assets.items():
        y = rets - df_gff['RF']
        model_k = regression_kernel(X, y)
        models[k] = model_k
    return models

def compute_factors_coeff(models):

    alpha = list()
    beta = list()
    for asset_k in models.keys():
        alpha_k, *beta_k = models[asset_k].params
        alpha.append(alpha_k)
        beta.append(beta_k)
    alpha = np.array(alpha).reshape((1, len(alpha)))
    beta = np.array(beta).T

    return alpha, beta

def factor_lin_model(X, alpha, beta):

    return X @ beta + alpha

def factor_lin_generator(alpha, beta):

    return partial(factor_lin_model, alpha=alpha, beta=beta)

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

FACTORS = {"famaFrench5Factor": ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
           "momentumFactor": ['MOM']
           }

PERCENTAGES = dict(largest=(0.05, 1),
                   large=(0.2, 2),
                   medium=(0.5, 3),
                   small=(0.2, 2),
                   smallest=(0.05, 1)
                   )
selected_tickers = select_assets(tickers_info, PERCENTAGES)

df_gff = get_factors(FACTORS.keys(), 'm')
df_gff =  trim_df_date(df_factors, start_date='1999-12-31', end_date='2021-01-31')
factor_names = get_factor_names(FACTORS)
factor_models = factors_regression(factor_names, df_gff,
                                   df_assets, regression_kernel=regression_OLS)
alpha, beta = compute_factors_coeff(factor_models)
factor_model = factor_lin_generator(alpha, beta)


# # ### read data
# ticker = 'msft'
# start = '2016-8-31'
# end = '2021-8-31'

# stock_data = yf.download(ticker, start, end)

# ff3_monthly = gff.famaFrench3Factor(frequency='m')
# ff5_monthly = gff.famaFrench5Factor(frequency='m')
# #momentum_monthly = gff.momentumFactor(frequency='m')

# ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
# ff3_monthly.set_index('Date', inplace=True)

# stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
# stock_returns.name = "Month_Rtn"
# ff_data = ff3_monthly.merge(stock_returns,on='Date')

# #### calculate betas
# X = ff_data[['Mkt-RF', 'SMB', 'HML']]
# y = ff_data['Month_Rtn'] - ff_data['RF']
# #X = sm.add_constant(X)
# ff_model = sm.OLS(y, X).fit()
# print(ff_model.summary())
# intercept, b1, b2, b3 = ff_model.params

# ####### expected 
# rf = ff_data['RF'].mean()
# market_premium = ff3_monthly['Mkt-RF'].mean()
# size_premium = ff3_monthly['SMB'].mean()
# value_premium = ff3_monthly['HML'].mean()

# expected_monthly_return = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium 
# expected_yearly_return = expected_monthly_return * 12
# print("Expected yearly return: " + str(expected_yearly_return))


# X = ff_data[['Mkt-RF', 'SMB', 'HML']]
# y = ff_data['Month_Rtn'] - ff_data['RF']
# X = sm.add_constant(X)
# ff_model = sm.OLS(y, X).fit()
# print(ff_model.summary())
# intercept, b1, b2, b3 = ff_model.params


# ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
# ff3_monthly.set_index('Date', inplace=True)


# tickers_list = ["aapl", "goog", "amzn", "BAC", "BA"] # example list
# tickers_data= {} # empty dictiona
# for ticker in tickers_list:
#     ticker_object = yf.Ticker(ticker)

#     #convert info() output from dictionary to dataframe
#     temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
#     temp.reset_index(inplace=True)
#     temp.columns = ["Attribute", "Recent"]
    
#     # add (ticker, dataframe) to main dictionary
#     tickers_data[ticker] = temp

# combined_data = pd.concat(tickers_data)
# combined_data = combined_data.reset_index()

# del combined_data["level_1"] # clean up unnecessary column
# #combined_data.columns = ["Ticker", "Attribute", "Recent"] # update column names

# employees = combined_data[combined_data["Attribute"]=="fullTimeEmployees"].reset_index()
# del employees["index"] # clean up unnecessary column

# employees

# index_weblist = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# tickers_sp500 = scrap_tickers_index(index_weblist)
# info_sp500 = ["marketCap", "sector"]
# tickers_info = get_tickers_info(tickers_sp500, info_sp500)
# tickers_info.dropna(inplace=True)
# tickers_info.sort_values('marketCap',ascending=False, inplace=True)

