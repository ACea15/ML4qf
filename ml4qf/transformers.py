import sklearn.preprocessing
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
            transformer_ins = Transformer(k, v['settings'])
        else:
            transformer_ins = Transformer(k)
        transformer = transformer_ins.transformer
        features_list = []
        for fi in v['features']:
            for ci in df.columns:
                if fi in ci[:len(fi)]:
                    features_list.append(ci)
        column_transformers.append((name, transformer, features_list))
    return column_transformers
