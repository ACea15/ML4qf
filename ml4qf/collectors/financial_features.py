"""sfdsfd."""
import numpy as np
import os
import pandas as pd
from datetime import timedelta
from datetime import date
import pandas_datareader as pdr
from collections import defaultdict
import yfinance as yf

class FinancialData:

    def __init__(self, inputs, df=None, **kwargs):

        if df is None:
            self.df = self.get_data(inputs.TICKER, inputs.YEAR0,
                                    inputs.MONTH0, inputs.DAY0, inputs.NUM_DAYS)
            self.df['price'] = self.df[inputs.PRICE]
            self.add_returns(inputs.LOG_RETURN)
        else:
            self.df = df
        if len(inputs.FEATURES) > 0:
            self.features = FeaturesPrice(self.df, **inputs.FEATURES)

    def add_returns(self, log_return):
        """Add the returns to a data frame with prices."""
        if not log_return:
            self.df['return'] = self.df['price'].pct_change()
        else:
            self.df['return'] = np.log(self.df['price'] /
                                       self.df['price'].shift(periods=1))

    @classmethod
    def clone(cls, inputs, df):
        return cls(inputs, df.copy())

    @staticmethod
    def get_data(ticker, year, month, day, num_days):
        """


        Parameters
        ----------
        ticker :

        year :

        month :

        day :

        num_days :


        Returns
        -------
        out :

        """
        date_end = date(year, month, day)
        date_start = date_end - timedelta(**{'days':num_days})
        data_folder = os.getcwd()+'/data'
        data_file = data_folder + f"/{ticker}_{date_start}_{date_end}"

        if not os.path.isdir(data_folder):
            print("***** Creating data folder *****")
            os.makedirs(data_folder)
        if os.path.isfile(data_file):
            print("***** Loading data from csv file *****")
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        else:
            print("***** Loading data from Yahoo Finance *****")
            df = yf.download(ticker, start=date_start, end=date_end)
            df.to_csv(data_file)
        return df


class FeaturesPrice:
    """Creates financial features using arbitrary rolling windows."""

    def __init__(self, df: pd.DataFrame, **features: dict):
        self.df = df.copy()
        self.features_in = features.keys()
        self.features_list = []
        self.get_features(**features)
        self.drop_nonfeatures()
        self.drop_nans()

    def get_features(self, **kwargs):
        """
        Below code features to an existing data frame
        """

        for k, v in kwargs.items():
            if (v == None or v == ''):
                self.add_feature(k)
            elif isinstance(v, (list, np.ndarray)):
                for vi in v:
                    self.add_feature(k, vi)
            elif isinstance(v, (int, float)):
                self.add_feature(k, v)
            else:
                raise ValueError('Invalid value for feature')


    def drop_nans(self):
        """Drop NaN values in the dataframe."""
        self.df.dropna(inplace=True)

    def drop_nonfeatures(self):
        self.df = self.df[self.features_list]

    def add_feature(self, label, period=None):
        _type = getattr(self, label)
        if period is None:
            self.df[f"{label}"] = _type(self.df, period)
            self.features_list.append(f"{label}")
        else:
            self.df[f"{label}{period}d"] = _type(self.df, period)
            self.features_list.append(f"{label}{period}d")

    def return_(self, df, period):

        out = df['return'].shift(periods=period)
        return out

    def Ret_(self, df, period):

        out = df['return'].rolling(period).sum()
        return out

    def Std_(self, df, period):

        out = df['return'].rolling(period).std()
        return out

    def momentum_(self, df, period):

        out = df['price'] - df['price'].shift(periods=period)
        return out

    def MA_(self, df, period):

        out = df['price'].rolling(period).mean()
        return out

    def EMA_(self, df, period):

        out = df['price'].ewm(period, adjust=False).mean()
        return out

    def OC_(self, df, period=None):

        out = df.Open - df.Close
        return out

    def HL_(self, df, period=None):

        out = df.High - df.Low
        return out

    def sign_return_(self, df, period=None):

        if period is None:
            return_var = 'return'
        else:
            return_var = 'return_{}d'.format(period)
            out = df[return_var].apply(lambda x: 0 if x<0 else 1)
        return out

    def sign_momentum_(self, df, period):

        momentum_var = 'momentum_{}d'.format(period)
        out = df[momentum_var].apply(lambda x: 0 if x<0 else 1)
        return out

def features_label(features_list):

    features_dict = defaultdict(list)
    for _ in features_list:
        feature_= _.split('_')
        feature = '_'.join(feature_[:-1])
        di = feature_[-1]
        if di != '':
            features_dict[feature + '_'].append(int(di[:-1]))
        else:
            features_dict[feature + '_'] = None
    return features_dict
