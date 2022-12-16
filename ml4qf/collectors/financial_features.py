"""sfdsfd."""
import numpy as np
import os
import pandas as pd
from datetime import datetime, timedelta
from datetime import date
import pandas_datareader as pdr
from collections import defaultdict

def add_returns(self, df, price='Close', log_return=False, *args, **kwargs):
    """
    Add the returns to a data frame with prices.
    """
    #self.df.rename(columns={price: 'price'})
    if not log_return:
        df['return'] = df[price].pct_change()
    else:
        df['return'] = np.log(df['price'] /
                              df['price'].shift(periods=1))


class FeaturesPrice:
    """
    Creates financial features using arbitrary rolling windows.
    """
    
    def __init__(self, df: pd.DataFrame, features: dict = None):
        self.df = df
        self.features_in = features
        self.features_list = []
        if features is not None:
            self.get_features(**features)
        
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
