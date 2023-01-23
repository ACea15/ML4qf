##############################################################################
# Add ml4qf library to python path.                                          #
#                                                                            #
# Preferable to do export PYTHONPATH="$PYTHONPATH:{path}" from command line  #
##############################################################################

import sys
import pathlib
file_path = sys.path[1]
sys.path.append(file_path + "/../")

################################
# Import libraries and modules #
################################

import numpy as np
import pathlib
import yfinance as yf
# from umap import UMAP
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.rcParams['figure.dpi'] = 80

import minisom
import umap
from sklearn.model_selection import train_test_split
import sklearn.metrics
import scipy.optimize
import sklearn.compose
import sklearn.pipeline
from scikeras.wrappers import KerasClassifier
import tensorflow.keras.utils
import plotly.express as px
import scipy.stats
import pandas as pd
import pandas_ta as ta

import ml4qf
import ml4qf.utils
import ml4qf.predictors.model_som
import ml4qf.collectors.financial_features
import ml4qf.transformers
import ml4qf.predictors.model_keras as model_keras
import ml4qf.predictors.model_tuning

#######################################
# Set random seed for keras and numpy #
#######################################
tensorflow.keras.utils.set_random_seed(42)

# 
###########################################
# Get data and create Features            #
# Ticker: EADSY (Airbus)                  #
# Ten years from October 2019 backwards   #
###########################################

FEATURES1 = {'Month_': None,
    'momentum_': [1, 2, 5, 8, 15, 23, 30, 40, 65, 75],
    'OC_': None,
    'HL_': None,
    'Ret_': [1, 5, 15, 20, 25, 30, 40, 50, 60, 75],
    #'RetLog_': list(range(10, 90, 10)),
    'Std_': list(range(3, 75, 5)),
    'MA_': [5, 10, 18, 25, 35, 50, 60, 70],
    'EMA_': [5, 10, 18, 25, 35, 50, 60, 70],
    'sign_return_': list(range(1,10)),
    'volume_':None,
    "ema": {"close": "volume", "length": [5, 10, 15, 20, 30], "prefix": "VOLUME"},
    "log_return": {"length": [1, 10, 20, 35, 50], "cumulative": True},
    "bias":{"length": [5, 10, 15, 20, 35, 50, 60, 75]},
    "aroon":{"length": [5, 10, 15, 20, 35, 50, 60, 75]},
    "stoch":{},
    "trix":{"length": [5, 10, 15, 20, 35, 50, 60, 75]},
    "tsi":{},
    #"volume---":{},
    "efi":{},
    "mfi": {"length": [5, 10, 15, 20, 35, 50, 60, 75]},
    "cmf":{"length": [5, 10, 15, 20, 35, 50, 60, 75]},
    "ohlc4":{},
    "volatility":"whole"
    #"momentum":"whole"
    #"percent_return": dict(length=2, append=False),
    #"log_return": [1, 2]
}

# 
###########################################
# Get data and create Features            #
# Ticker: EADSY (Airbus)                  #
# Ten years from October 2019 backwards   #
###########################################

data = ml4qf.collectors.financial_features.FinancialData("EADSY", 2019, 10, 1, 365*16, FEATURES1)
img_dir = "./img/" + data.label
pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
df_  = data.features.df.drop(data.df.columns, axis=1)
df_.dropna(inplace=True)

own_features = list(df_.columns[:df_.columns.get_loc('volume_')+1])
pa_features = list(df_.columns[df_.columns.get_loc('volume_') + 1:df_.columns.get_loc('OHLC4')+1])
pa_volfeatures = list(df_.columns[df_.columns.get_loc('OHLC4')+1:])
total_features = len(own_features) + len(pa_features) + len(pa_volfeatures)
print("######################")
print(own_features)
print("######################")
print(pa_features)
print("######################")
print(pa_volfeatures)
print("######################")
assert total_features == len(df_.columns), "Number of features not matching in dataframe"

fig1_path= img_dir +'/stock_Close.png'
fig1 = px.line(df_, y=['momentum_1d', 'momentum_5d', 'momentum_15d'])
fig1.write_image(fig1_path)
fig1_path

fig1_path= img_dir +'/correlation.png'
df_corr = df_.corr().round(2)
fig1 = px.imshow(np.abs(df_corr))
fig1.layout.height = 600
fig1.layout.width = 600
fig1.write_image(fig1_path)
fig1_path

alpha_min = scipy.optimize.bisect(ml4qf.utils.fix_imbalance, -0.01, 0.01, args=(data, df_.index))
df_['target'] = np.where(data.df.loc[df_.index]['returns'].shift(-1) > alpha_min, 1, 0)
df_.target.value_counts()

zscores = np.abs(scipy.stats.zscore(df_)).max()
print(zscores)

transformers = {'SeasonTransformer':{'features': ['Month_']},
                'MinMaxScaler': {'features': ['sign_return']},
                'StandardScaler_1': {'features': ['EMA', 'MA', 'Std', 'Ret', 'OC']},
                'RobustScaler': {'features': ['momentum', 'volume', 'HL']},
                'StandardScaler_2': {'features': pa_volfeatures},
                'StandardScaler_3': {'features': pa_features}
                }

columns = ml4qf.transformers.build_transformation(df_, transformers)
columns_validation = ml4qf.transformers.build_transformation(df_, transformers)
ct = sklearn.compose.ColumnTransformer(columns, remainder='passthrough')
#ct_validation = sklearn.compose.ColumnTransformer(columns, remainder='passthrough')

Xtrain, Xtest = train_test_split(df_.to_numpy(), train_size=0.8, shuffle=False)
len_train = len(Xtrain)
len_test = len(Xtest)
df_train = df_.iloc[:len_train, :]
df_test = df_.iloc[len_train:, :]
Xtrain_scaled = ct.fit_transform(df_train)
Xtrain_scaled = ml4qf.transformers.swap_features(Xtrain_scaled, df_train, ct)
Xtest_scaled = ct.transform(df_test)
Xtest_scaled = ml4qf.transformers.swap_features(Xtest_scaled, df_test, ct)
df_train_scaled = ml4qf.transformers.scale_df(df_train, columns_validation)
assert (Xtrain_scaled == df_train_scaled.to_numpy()).all(), "scaling failed"
#Xtrain_scaled = ct.transform(Xtrain)

som_labels = None
#som_labels = ['volume_', 'THERMOma_20_2_0.5', 'momentum_1d', 'VOLUME_EMA_20', 'VOLUME_EMA_10', 'Std_73d', 'Std_48d', 'TRUERANGE_1', 'sign_return_1d', 'EMA_5d', 'TRIX_20_9', 'Month_', 'target', 'Std_58d', 'Std_8d', 'HWL', 'sign_return_7d', 'NATR_14', 'Std_28d', 'momentum_23d', 'TRIXs_50_9', 'BBB_5_2.0', 'sign_return_4d', 'Std_38d', 'TRIX_35_9', 'THERMOl_20_2_0.5', 'sign_return_3d', 'AROOND_35', 'VOLUME_EMA_15', 'VOLUME_EMA_5', 'THERMO_20_2_0.5']
if som_labels is None:
  som_size = 50
  som_obj = ml4qf.predictors.model_som.Model(som_size, som_size, Xtrain_scaled, sigma=1.5, learning_rate=0.1, 
                                             neighborhood_function='gaussian', num_iter=10000, random_seed=42)
  som_labels = som_obj.iterate_som_selection(min_num_features=30, labels=list(df_train.columns), a_range=[0.01, 0.03, 0.05, 0.08, 0.1, 0.2], num_iterations=30)
print(som_labels)

# for i, f in enumerate(feature_names):
#     plt.subplot(3, 3, i+1)
#     plt.title(f)
#     plt.pcolor(W[:,:,i].T, cmap='coolwarm')
#     plt.xticks(np.arange(size+1))
#     plt.yticks(np.arange(size+1))
# plt.tight_layout()
# plt.show()
fig1_path= img_dir +'/som.png'
fig1 = px.imshow(som_obj.W[:20,30:,20].T)
fig1.layout.height = 1000
fig1.layout.width = 1000
fig1.write_image(fig1_path)
fig1_path

index_reducedlabels = [df_train.columns.get_loc(i) for i in som_labels]
dftrain_reduced = df_train[som_labels]
dftest_reduced = df_test[som_labels]
assert (dftrain_reduced.to_numpy() == Xtrain[:, index_reducedlabels]).all(), "Reduced matrix not maching dimensions"
Xtrain_reduced = Xtrain_scaled[:, index_reducedlabels]
Xtest_reduced = Xtest_scaled[:, index_reducedlabels]
#Xtest_reduced = Xtest_scaled[:, index_reducedlabels]

SEQ_LEN = 15
y_train = df_train['target'].to_numpy()
y_test  = df_test.target.to_numpy()

lstm_model = model_keras.Model_binary(keras_model='Sequential',
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy',
                                      metrics=['accuracy','binary_accuracy'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)

searcher_name = 'GridSearchCV'
layers_hyper = []
###########
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=100, activation = 'relu', name='LSTM1')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['LSTM_2'] = dict(units=50, activation = 'elu', return_sequences=False, name='LSTM2')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=70, activation = 'relu', return_sequences=True, name='LSTM1')
layers_dict['Dropout_1'] = dict(rate=0.5, name='Drouput1')
layers_dict['LSTM_2'] = dict(units=50, activation = 'relu', return_sequences=False, name='LSTM2')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
############
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['LSTM_2'] = dict(units=40, activation = 'relu', return_sequences=True, name='LSTM2')
layers_dict['LSTM_3'] = dict(units=30, activation = 'elu', return_sequences=False, name='LSTM3')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['Dropout_1'] = dict(rate=0.35, name='Drouput1')
layers_dict['LSTM_2'] = dict(units=25, activation = 'elu', return_sequences=True, name='LSTM2')
layers_dict['Dropout_2'] = dict(rate=0.35, name='Drouput2')
layers_dict['LSTM_3'] = dict(units=25, activation = 'elu', return_sequences=False, name='LSTM3')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################

###########
hyper_grid = {'seqlen':[5, 15, 30, 45, 75],
              'layers':layers_hyper,
              'optimizer_name':['adam', 'adamax'],
              'batch_size': [8, 16, 32,64,128]
              }
searcher_settings = {#'scoring':'f1',
                     #'n_iter':25,
                     'n_jobs':8,
                     'verbose': False}
cv_name = 'TimeSeriesSplit'
cv_settings = {'n_splits': 2}
_hypertuning1 = ml4qf.predictors.model_tuning.HyperTuning(lstm_model, searcher_name, searcher_settings,
                                                          hyper_grid, cv_name, cv_settings)
hypertuning1 = _hypertuning1()
hypertuning1.fit(Xtrain_reduced, y_train, epochs=85, verbose=False, shuffle=False)

import pickle
pickle.dump(hypertuning1, "./data/hypertuning1.pickle")
with open("./data/hypertuning1.pickle", 'wb') as handle:
    pickle.dump(hypertuning1, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
