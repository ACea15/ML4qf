"""sfdsfd."""
__all__ = ["FinancialData", "FeaturesPrice"]
import numpy as np
import os
import pandas as pd
from datetime import timedelta
from datetime import date
from collections import defaultdict
import yfinance as yf
import pandas_ta as ta
import pathlib
import ml4qf.utils

def get_ticker_dates(year0=None,
                     month0=None,
                     day0=None,
                     year1=None,
                     month1=None,
                     day1=None,
                     num_days=None
                     ):
    
    if (bool(year0) & bool(month0) & bool(day0) & bool(num_days)):
        date_start = date(year0, month0, day0)
        date_end = date_start + timedelta(**{"days": num_days})
    elif (bool(year1) & bool(month1) & bool(day1) & bool(num_days)):
        date_end = date(year1, month1, day1)
        date_start = date_end - timedelta(**{"days": num_days})
    elif (bool(year0) & bool(month0) & bool(day0) &
          bool(year1) & bool(month1) & bool(day1)):
        date_end = date(year1, month1, day1)
        date_start = date(year0, month0, day0)
    else:
        raise ValueError("Wrong dates input to get_data")
    return date_start, date_end

class APIs_get_data:

    @staticmethod
    def data_yf_download(ticker,
                         date_start,
                         date_end,
                         interval,
                         *args, **kwargs):

        df = yf.download(
            ticker,
            start=date_start,
            end=date_end,
            interval=interval,
            ignore_tz=True)
        print("***** Loading data from Yahoo Finance *****")
        return df

class FinancialDataContainer:
    
    def __init__(
            self,
            TICKERS: list[str],
            START_DATE,
            END_DATE,
            INTERVAL='1d',
            DATA_FOLDER=None,
            FEATURES=None,
            PRICE="Close",
            *args,
            **kwargs):
        
        for ti in TICKERS:
            financial_data = FinancialData(ti,
                                           START_DATE,
                                           END_DATE,
                                           INTERVAL,
                                           DATA_FOLDER,
                                           FEATURES,
                                           PRICE
                                           )
            setattr(self,
                    ml4qf.utils.clean(ti),
                    financial_data
                    )
    
class FinancialData:
    
    def __init__(
            self,
            TICKER,
            START_DATE,
            END_DATE,
            INTERVAL='1d',
            DATA_FOLDER=None,
            FEATURES=None,
            PRICE="Close",
            df=None,
            *args,
            **kwargs):
        
        if df is None:
            self.df = self.get_data(TICKER,
                                    START_DATE,
                                    END_DATE,
                                    INTERVAL,
                                    DATA_FOLDER)
            self.df["price"] = self.df[PRICE]
            self.add_returns()
        else:
            self.df = df
        if FEATURES is not None:
            self.features = FeaturesPrice(self.df, **FEATURES)

    def add_returns(self, log_return=True):
        """Add the returns to a data frame with prices."""
        self.df["returns"] = self.df["price"].pct_change()
        if log_return:
            self.df["return_log"] = np.log(
                self.df["price"] / self.df["price"].shift(periods=1)
            )

    @classmethod
    def clone(cls, inputs, df):
        return cls(inputs, df.copy())

    @staticmethod
    def get_data(ticker,
                 date_start,
                 date_end,
                 interval='1d',
                 data_folder=None,
                 api_name="yf_download",
                 save_data=True,
                 *args, **kwargs):

        if data_folder is not None:
            folder_path = pathlib.Path(data_folder)
            if not folder_path.is_dir():
                print("***** Creating data folder *****")
                folder_path.mkdir(parents=True, exist_ok=True)
            data_file = folder_path / f"{ticker}_{date_start}_{date_end}"
            if data_file.is_file():
                print("***** Loading data from csv file *****")
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                return df
        else:
            api = getattr(APIs_get_data, f"data_{api_name}")
            df = api(
                ticker=ticker,
                date_start=date_start,
                date_end=date_end,
                interval=interval,
                *args, **kwargs
            )
            return df
        if save_data:
            df.to_csv(data_file)
        return df

class FeaturesPrice:
    """Creates financial features using arbitrary rolling windows."""

    def __init__(self, df: pd.DataFrame, strategy_name="strategy_1", **features: dict):
        self.df = df.copy()
        self.features_in = features.keys()
        self.features_list = []
        self.ta_strategy = dict()
        self.get_features(**features)
        if len(self.ta_strategy) > 0:
            self.ta, self.whole_ta = self.build_ta(self.ta_strategy)
            self.my_strategy = ta.Strategy(name=strategy_name, ta=self.ta)
            self.df.ta.strategy(self.my_strategy)
            if len(self.whole_ta) > 0:
                for wtai in self.whole_ta:
                    self.df.ta.strategy(wtai)
        # self.drop_nonfeatures()
        # self.drop_nans()

    def get_features(self, **kwargs):
        """
        Below code features to an existing data frame
        """

        for k, v in kwargs.items():
            if hasattr(self, k):
                if v == None or v == "":
                    self.add_feature(k)
                elif isinstance(v, (list, np.ndarray)):
                    for vi in v:
                        self.add_feature(k, vi)
                elif isinstance(v, (int, float)):
                    self.add_feature(k, v)
                else:
                    raise ValueError("Invalid value for feature")
            else:
                self.ta_strategy[k] = v

    @staticmethod
    def build_ta(ta_dict):
        ta = []
        whole_ta = []
        for k, v in ta_dict.items():
            if isinstance(v, int):
                item = dict(kind=k, length=v)
                ta.append(item)
            elif isinstance(v, list):
                for i in v:
                    item = dict(kind=k, length=i)
                    ta.append(item)
            elif isinstance(v, dict):
                if "length" in v.keys() and isinstance(v["length"], list):
                    lengths = v.pop("length")
                    for li in lengths:
                        item = dict(kind=k, length=li, **v)
                        ta.append(item)
                else:
                    item = dict(kind=k, **v)
                    ta.append(item)
            elif isinstance(v, str) and v == "whole":
                whole_ta.append(k)
            else:
                raise ValueError("Invalid ta entry")
        return ta, whole_ta

    def drop_nans(self):
        """Drop NaN values in the dataframe."""
        self.df.dropna(inplace=True)

    def drop_nonfeatures(self):
        self.df = self.df[self.features_list]

    def add_feature(self, label, value=None):
        _type = getattr(self, label)
        if value is None:
            self.df[f"{label}"] = _type(self.df, value)
            self.features_list.append(f"{label}")
        else:
            self.df[f"{label}{value}d"] = _type(self.df, value)
            self.features_list.append(f"{label}{value}d")

    def RetLog_(self, df, period):
        out = df["return_log"].rolling(period).sum()
        return out

    def Ret_(self, df, period):
        out = (
            df["price"]
            .rolling(period + 1)
            .apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
        )
        return out

    def Std_(self, df, period):
        out = df["returns"].rolling(period).std()
        return out

    def momentum_(self, df, period):
        out = df["price"] - df["price"].shift(periods=period)
        return out

    def MA_(self, df, period):
        out = df["price"].rolling(period + 1).mean()
        return out

    def EMA_(self, df, period):
        out = df["price"].ewm(period + 1, adjust=False).mean()
        return out

    def OC_(self, df, period=None):
        out = df.Open - df.Close
        return out

    def HL_(self, df, period=None):
        out = df.High - df.Low
        return out

    def sign_return_(self, df, period=None):
        out = (
            df["returns"]
            .rolling(period + 1)
            .apply(lambda x: 0 if (x[-1] - x[0]) / x[0] < 0 else 1)
        )
        return out

    def volume_(self, df, period=None):
        return df.Volume

    def Month_(self, df, period=None):
        out = df.index.month
        return out


def features_label(features_list):
    features_dict = defaultdict(list)
    for _ in features_list:
        feature_ = _.split("_")
        feature = "_".join(feature_[:-1])
        di = feature_[-1]
        if di != "":
            features_dict[feature + "_"].append(int(di[:-1]))
        else:
            features_dict[feature + "_"] = None
    return features_dict


if __name__ == "__main__":
    FEATURES1 = {  #'momentum_': [1],
        #'OC_': None,
        #'HL_': None,
        "Ret_": [1, 2],
        "RetLog_": [1, 2],
        #'Std_': list(range(10, 90, 10)),
        #'MA_': [5, 10, 25, 50],
        #'EMA_': [5, 10, 25, 50],
        #'sign_return_': [1, 2, 5, 8, 15, 23],
        "volume_": None,
        "percent_return": [1, 2],
        # "percent_return": dict(length=2, append=False),
        "log_return": [1, 2],
    }

    # ticker = ml4qf.inputs.Input_ticker("EADSY", 2019, 10, 1, 5, FEATURES1)
    data = FinancialData("EADSY", 2019, 10, 1, 365, FEATURES1)
