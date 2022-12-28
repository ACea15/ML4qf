import ml4qf.inputs
import ml4qf.collectors.financial_features as financial_features


FEATURES1 = {'return_': [1, 2, 5, 8, 15, 23],
            'momentum_': [1, 2, 5, 8, 15, 23],
            'OC_': None,
            'HL_': None,
            'Ret_': list(range(10, 90, 10)),
            'Std_': list(range(10, 90, 10)),
            'MA_': [5, 10, 25, 50],
            'EMA_': [5, 10, 25, 50],
            'sign_return_': [1, 2, 5, 8, 15, 23],
            'sign_momentum_':[1, 2, 5, 8, 15, 23]
            }


ticker = ml4qf.inputs.Input_ticker('BSP.F', 2022, 1, 1, 10, FEATURES1)
data = financial_features.FinancialData(ticker)
FEATURES2 = {'return_': [1, 15, 23],
            'momentum_': [1, 23],
            'OC_': None,
            'HL_': None,
            }

ticker2 = ticker.clone(FEATURES=FEATURES2)
data2 = data.clone(ticker2, data.df)
#features = financial_features.FeaturesPrice(data.df, **ticker.FEATURES)

# import yfinance as yf
# data = yf.download("SPY AAPL", start="2017-01-01", end="2017-04-30")

layers_dict = dict()
layers_dict['LSTM'] = dict(units=5, activation = 'relu', return_sequences=False, name='LSTM')
layers_dict['Dense'] = dict(units=1, name='Output')

def dict2tuple(x: dict) -> tuple:

    y = []
    for k, v in x.items():
        z = []
        z.append(k)
        if isinstance(v, dict):
            z.append(dict2tuple(v))
        else:
            z.append(v)
        y.append(tuple(z))
    return tuple(y)

layers = (('LSTM',
           ('units', 5),
           ('activation', 'relu'),
           ('return_sequences', False),
           ('name', 'LSTM')
           ),
          ('Dense',
           ('units',1),
           ('name', 'Output')
           )
          )
