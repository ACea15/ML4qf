import ml4qf.utils
###########################################
# Get data and create Features            #
# Ticker: EADSY (Airbus)                  #
# Ten years from October 2019 backwards   #
###########################################
RANDOM_SEED = 7

#################
# LSTM SETTINGS #
#################

layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=4, name='LSTM1')
layers_dict['Dense_1'] = dict(units=1, name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)

lstm = dict(keras_model='Sequential',
            layers=layers_tuple,
            seqlen=1,
            optimizer_name='adam',
            loss_name='mean_squared_error',
            metrics=['accuracy','binary_accuracy', 'mse'],
            optimizer_sett=None,
            compile_sett=None,
            loss_sett=None,
            batch_size=1)

TRAIN_SIZE = 0.67

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from sklearn.compose import make_column_selector
    import numpy as np
    import pandas as pd  
    X = pd.DataFrame({'city': ['London', 'London', 'Paris', 'Sallisaw'],
                      'rating': [5, 3, 4, 5]})  
    ct = make_column_transformer(
          (StandardScaler(),
           make_column_selector(dtype_include=np.number)),  # rating
          (OneHotEncoder(),
           make_column_selector(dtype_include=object)))  # city
    ct.fit_transform(X)  

    X = pd.DataFrame({'city_4': ['London', 'London', 'Paris', 'Sallisaw'],
                      'city_2': [1, 3, 4, 5],
                      'rating': [5, 3, 4, 5]})

    make_column_selector(pattern='city')(X)
