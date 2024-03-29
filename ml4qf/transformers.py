import sklearn.preprocessing
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.compose import make_column_selector, ColumnTransformer

class TransformerFactory:

    def __init__(self,  transformer_name, transformer_settings=None):

        self.transformer_name = transformer_name
        self.transformer_settings = transformer_settings
        if self.transformer_settings == None:
            self.transformer_settings = dict()
        self.transformer = None
        self.set_transformer()

    def set_transformer(self):

        try: # transformer in this module
            self.transformer_type = globals()[self.transformer_name]
            self.transformer = self.transformer_type(**self.transformer_settings)
        except KeyError: # transformer in sklearn
            self.transformer_type = getattr(sklearn.preprocessing, self.transformer_name)
            self.transformer = self.transformer_type(**self.transformer_settings)

    def __call__(self):

        return self.transformer
    
# create custom time transformer 
class SeasonTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):

    def fit(self, X, y=None):
        self.data = pd.DataFrame(
            {
        'Months': ["January",
                   "February",
                   "March",
                   "April",
                   "May",
                   "June",
                   "July",
                   "August",
                   "September",
                   "October",
                   "November",
                   "December"]
            }
        )
        self.num_months = 12
        return self
       
    def transform(self, X):
        Xt = X.copy()
        Xt = np.sin(2 * np.pi * Xt / self.num_months)        
        return Xt

class Build_ColumnTransformer:
    
    def __init__(self, df: pd.DataFrame, transformers: dict):
        self.df = df
        self.transformers = transformers
        self.build()
    
    def build(self):

        self.column_transformers = []
        for k, v in self.transformers.items():
            name = k.replace('Scaler', '')
            if 'settings' in v.keys():
                transformer = TransformerFactory(k.split("_")[0], v['settings'])()
            else:
                transformer = TransformerFactory(k.split("_")[0])()
            features_list = []
            for fi in v['features']:
                # for ci in df.columns:
                #     if (fi + "_") in ci[:len(fi)+1] or fi == ci:
                #         features_list.append(ci)
                features_list += make_column_selector(pattern=fi)(self.df)
            self.column_transformers.append((name, transformer, features_list))

    def __call__(self, **kwargs):

        return ColumnTransformer(self.column_transformers, **kwargs)

def scale_df(df, transformers):
    df_2 = df.copy()
    sizeX = len(df_2)
    for ci in transformers:
        for fi in ci[2]:
          df_2[fi] = ci[1].fit_transform(np.array(df[fi]).reshape(sizeX,1)).reshape(sizeX)
    return df_2

def swap_features(X, df, ct):
    X_new = X.copy()
    for i, fi in enumerate(ct.get_feature_names_out()):
        j = df.columns.get_loc(fi.split("__")[1])
        if i != j:
            X_new[:, [j]] = X[:, [i]]
    return X_new

def find_features(df, ct):
    dict1 = {}
    for i, fi in enumerate(ct.get_feature_names_out()):
        j = df.columns.get_loc(fi.split("__")[1])
        dict1[fi.split("__")[1]] = [i, j]
    return dict1

def check_inversetransform():
    pass
