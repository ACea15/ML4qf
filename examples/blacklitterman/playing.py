import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff

### read data
ticker = 'msft'
start = '2016-8-31'
end = '2021-8-31'

stock_data = yf.download(ticker, start, end)

ff3_monthly = gff.famaFrench3Factor(frequency='m')
#ff5_monthly = gff.famaFrench5Factor(frequency='m')
#momentum_monthly = gff.momentumFactor(frequency='m')

ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff3_monthly.set_index('Date', inplace=True)

stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
stock_returns.name = "Month_Rtn"
ff_data = ff3_monthly.merge(stock_returns,on='Date')

#### calculate betas
X = ff_data[['Mkt-RF', 'SMB', 'HML']]
y = ff_data['Month_Rtn'] - ff_data['RF']
X = sm.add_constant(X)
ff_model = sm.OLS(y, X).fit()
print(ff_model.summary())
intercept, b1, b2, b3 = ff_model.params

####### expected 
rf = ff_data['RF'].mean()
market_premium = ff3_monthly['Mkt-RF'].mean()
size_premium = ff3_monthly['SMB'].mean()
value_premium = ff3_monthly['HML'].mean()

expected_monthly_return = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium 
expected_yearly_return = expected_monthly_return * 12
print("Expected yearly return: " + str(expected_yearly_return))




tickers_list = ["aapl", "goog", "amzn", "BAC", "BA"] # example list
tickers_data= {} # empty dictiona
for ticker in tickers_list:
    ticker_object = yf.Ticker(ticker)

    #convert info() output from dictionary to dataframe
    temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
    temp.reset_index(inplace=True)
    temp.columns = ["Attribute", "Recent"]
    
    # add (ticker, dataframe) to main dictionary
    tickers_data[ticker] = temp

combined_data = pd.concat(tickers_data)
combined_data = combined_data.reset_index()

del combined_data["level_1"] # clean up unnecessary column
#combined_data.columns = ["Ticker", "Attribute", "Recent"] # update column names

employees = combined_data[combined_data["Attribute"]=="fullTimeEmployees"].reset_index()
del employees["index"] # clean up unnecessary column

employees
