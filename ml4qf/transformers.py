import sklearn.preprocessing
import numpy as np

class Transformer:

    def __init__(self,  transformer_name, transformer_settings={}):


        self.transformer_name = transformer_name
        self.transformer_settings = transformer_settings
        self.transformer = None
        self.set_transformer()

    def set_transformer(self):

        self.transformer_type = getattr(sklearn.preprocessing, self.transformer_name)
        self.transformer = self.transformer_type(**self.transformer_settings)


def build_transformation(df, transformers):

    column_transformers = []
    for k, v in transformers.items():
        name = k.replace('Scaler', '')
        if 'settings' in v.keys():
            transformer_ins = Transformer(k.split("_")[0], v['settings'])
        else:
            transformer_ins = Transformer(k.split("_")[0])
        transformer = transformer_ins.transformer
        features_list = []
        for fi in v['features']:
            for ci in df.columns:
                if (fi + "_") in ci[:len(fi)+1] or fi == ci:
                    features_list.append(ci)
        column_transformers.append((name, transformer, features_list))
    return column_transformers

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
