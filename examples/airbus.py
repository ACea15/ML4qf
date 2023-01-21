import sys
import pathlib
file_path = sys.path[1]
sys.path.append(file_path + "/../")

import numpy as np
import pathlib
import yfinance as yf
import ml4qf
import ml4qf.utils
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

import plotly.express as px
import scipy.stats
import pandas as pd
import pandas_ta as ta
import ml4qf.inputs
import ml4qf.predictors.model_som
import ml4qf.collectors.financial_features
import ml4qf.transformers
import ml4qf.predictors.model_som
import ml4qf.predictors.model_keras as model_keras
import ml4qf.predictors.model_tuning

FEATURES1 = {'momentum_': [1, 2, 5, 8, 15, 23],
             'OC_': None,
             'HL_': None,
             'Ret_': [1, 5, 30, 35],
             #'RetLog_': list(range(10, 90, 10)),
             'Std_': list(range(10, 90, 10)),
             'MA_': [5, 10, 25, 50],
             'EMA_': [5, 10, 25, 50],
             'sign_return_': list(range(1,10)),
             'volume_':None,
             "ema": {"close": "volume", "length": [5, 10, 20], "prefix": "VOLUME"},
             "log_return": {"length": [1, 20], "cumulative": True},
             "ohlc4":{},
             "volatility":"whole"
             #"momentum":"whole"
             #"percent_return": dict(length=2, append=False),
             #"log_return": [1, 2]
             }

data = ml4qf.collectors.financial_features.FinancialData("EADSY", 2019, 10, 1, 365*10, FEATURES1)
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

ymin = scipy.optimize.bisect(ml4qf.utils.fix_imbalance, -0.01, 0.01, args=(data, df_.index))
df_['target'] = np.where(data.df.loc[df_.index]['returns'].shift(-1) > ymin, 1, 0)
df_.target.value_counts()

zscores = np.abs(scipy.stats.zscore(df_)).max()
print(zscores)

transformers = {'MinMaxScaler': {'features': ['sign_return']},
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

som_size = 50
som_obj = ml4qf.predictors.model_som.Model(som_size, som_size, Xtrain_scaled, sigma=1.5, learning_rate=0.1, 
neighborhood_function='gaussian', num_iter=10000, random_seed=42)
som_labels = som_obj.iterate_som_selection(min_num_features=30, labels=list(df_train.columns), a_range=[0.01, 0.03, 0.05, 0.08, 0.1, 0.2], num_iterations=30)
print(som_labels)

# # for i, f in enumerate(feature_names):
# #     plt.subplot(3, 3, i+1)
# #     plt.title(f)
# #     plt.pcolor(W[:,:,i].T, cmap='coolwarm')
# #     plt.xticks(np.arange(size+1))
# #     plt.yticks(np.arange(size+1))
# # plt.tight_layout()
# # plt.show()
# fig1_path= img_dir +'/som.png'
# fig1 = px.imshow(W[:,:,0].T)
# fig1.layout.height = 600
# fig1.layout.width = 600
# fig1.write_image(fig1_path)
# fig1_path

index_reducedlabels = [df_train.columns.get_loc(i) for i in som_labels]
dftrain_reduced = df_train[som_labels]
dftest_reduced = df_test[som_labels]
assert (dftrain_reduced.to_numpy() == Xtrain[:, index_reducedlabels]).all(), "Reduced matrix not maching dimensions"
Xtrain_reduced = Xtrain_scaled[:, index_reducedlabels]
Xtest_reduced = Xtest_scaled[:, index_reducedlabels]
#Xtest_reduced = Xtest_scaled[:, index_reducedlabels]

SEQ_LEN = 30
y_train = df_train['target'].to_numpy()

layers_dict = dict()
############
# layers_dict['LSTM'] = dict(units=5, activation = 'relu', return_sequences=False, name='LSTM')
# layers_dict['Dense'] = dict(units=1, name='Output')
############
# layers_dict['LSTM_1'] = dict(units=100*2, activation = 'elu', return_sequences=True, name='LSTM1')
# layers_dict['Dropout_1'] = dict(rate=0.4, name='Drouput1')
# layers_dict['LSTM_2'] = dict(units=100, activation = 'elu', return_sequences=True, name='LSTM2')
# layers_dict['Dropout_2'] = dict(rate=0.4, name='Drouput2')
# layers_dict['LSTM_3'] = dict(units=100, activation = 'elu', return_sequences=False, name='LSTM3')
# layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
#####################
############
# layers_dict['LSTM_1'] = dict(units=100, activation = 'elu', return_sequences=True, name='LSTM1')
# layers_dict['LSTM_2'] = dict(units=100, activation = 'elu', return_sequences=False, name='LSTM2')
# layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
#####################
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', name='LSTM1')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
#####################
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
#######################
base_model = model_keras.Model_binary(keras_model='Sequential', layers=layers_tuple,
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy',
                                      metrics=['accuracy','binary_accuracy', 'mse'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)
base_model.fit(Xtrain_reduced, y_train, epochs=70, shuffle=False, verbose=1)

# summary
base_model._model.summary()

y_test  = df_test.target.to_numpy()
ypred_basemodel = base_model.predict(Xtest_reduced, y_test).reshape(len(y_test[SEQ_LEN-1:]))
test_report = sklearn.metrics.classification_report(y_test[SEQ_LEN-1:], 
                                                    ypred_basemodel, output_dict=True)
dftest_report = pd.DataFrame(test_report).transpose()
print(dftest_report)

ypred_basemodeltrain = base_model.predict(Xtrain_reduced, y_train).reshape(len(y_train[SEQ_LEN-1:]))
train_report = sklearn.metrics.classification_report(y_train[SEQ_LEN-1:],
                                                     ypred_basemodeltrain, output_dict=True)
dftrain_report = pd.DataFrame(train_report).transpose()
print(dftrain_report)

umap_model = umap.UMAP(n_components=3)
layers_dict = dict()
#####################
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', name='LSTM1')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
#####################
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
#######################
lstm_model = model_keras.Model_binary(keras_model='Sequential', layers=layers_tuple,
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy',
                                      metrics=['accuracy','binary_accuracy', 'mse'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)

pipe = sklearn.pipeline.Pipeline([('umap', umap_model),
                                  ('lstm', lstm_model)])

pipe.fit(Xtrain_reduced, y_train, lstm__epochs=70, lstm__shuffle=False)

# summary

y_test  = df_test.target.to_numpy()
ypred_basemodel = pipe.predict(Xtest_reduced)#.reshape(len(y_test[SEQ_LEN-1:]))
test_report = sklearn.metrics.classification_report(y_test[SEQ_LEN-1:], 
                                                    ypred_basemodel, output_dict=True)
dftest_report = pd.DataFrame(test_report).transpose()
print(dftest_report)

ypred_basemodeltrain = pipe.predict(Xtrain_reduced)#.reshape(len(y_train[SEQ_LEN-1:]))
train_report = sklearn.metrics.classification_report(y_train[SEQ_LEN-1:],
                                                     ypred_basemodeltrain, output_dict=True)
dftrain_report = pd.DataFrame(train_report).transpose()
print(dftrain_report)

umap_model = umap.UMAP()
lstm_model = model_keras.Model_binary(keras_model='Sequential',
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy',
                                      metrics=['accuracy','binary_accuracy', 'mse'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)
pipe = sklearn.pipeline.Pipeline([
                                  ('lstm', lstm_model)])
pipe.get_params()

searcher_name = 'RandomizedSearchCV'
layers_hyper = []
###########
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=100, activation = 'elu', name='LSTM1')
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
############
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['Dropout_1'] = dict(rate=0.3, name='Drouput1')
layers_dict['LSTM_2'] = dict(units=25, activation = 'elu', return_sequences=True, name='LSTM2')
layers_dict['Dropout_2'] = dict(rate=0.3, name='Drouput2')
layers_dict['LSTM_3'] = dict(units=25, activation = 'elu', return_sequences=False, name='LSTM3')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################

###########
hyper_grid = {#'umap':dict(n_neighbors=[5, 15, 30, 50, 100],
              #            n_components=[3, 8, 15, 30],
              #            min_dist=[0.05, 0.1, 0.4, 0.75],
              #            random_state=42),
              #'umap__n_neighbors':[30, 50],    
              #'umap__n_components':[3, 2],         
              #'umap__min_dist':[0.05, 0.2],     
              #'umap__random_state':[42],                    
              'lstm__seqlen':[10, 25],
              'lstm__layers':layers_hyper,
              'lstm__optimizer_name':['adam']
              }
searcher_settings = {'scoring':'f1',
                     'n_iter':25,
                     'verbose': True}
cv_name = 'TimeSeriesSplit'
cv_settings = {'n_splits': 3}
_hypertuning1 = ml4qf.predictors.model_tuning.HyperTuning(pipe, searcher_name, searcher_settings,
                                                          hyper_grid, cv_name, cv_settings)
hypertuning1 = _hypertuning1()
hypertuning1.fit(Xtrain_reduced, y_train, lstm__epochs=70, lstm__shuffle=False)

import tensorflow.keras.backend
import itertools
umap_model = umap.UMAP()
lstm_model = model_keras.Model_binary(keras_model='Sequential',
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy',
                                      metrics=['accuracy','binary_accuracy', 'mse'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)
pipe = sklearn.pipeline.Pipeline([('umap', umap_model),
                                  ('lstm', lstm_model)])

searcher_name = 'RandomizedSearchCV'
layers_hyper = []
###########
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=100, activation = 'elu', name='LSTM1')
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
############
layers_dict = dict()
layers_dict['LSTM_1'] = dict(units=50, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['Dropout_1'] = dict(rate=0.3, name='Drouput1')
layers_dict['LSTM_2'] = dict(units=25, activation = 'elu', return_sequences=True, name='LSTM2')
layers_dict['Dropout_2'] = dict(rate=0.3, name='Drouput2')
layers_dict['LSTM_3'] = dict(units=25, activation = 'elu', return_sequences=False, name='LSTM3')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
layers_hyper.append(layers_tuple)
#####################
def product_dict(**kwargs):
  keys = kwargs.keys()
  vals = kwargs.values()
  for instance in itertools.product(*vals):
      yield dict(zip(keys, instance))

###########
hyper_grid = {#'umap':dict(n_neighbors=[5, 15, 30, 50, 100],
              #            n_components=[3, 8, 15, 30],
              #            min_dist=[0.05, 0.1, 0.4, 0.75],
              #            random_state=42),
              'umap__n_neighbors':[30],    
              'umap__n_components':[18],         
              'umap__min_dist':[0.05],     
              'umap__random_state':[42],                    
              #'lstm__seqlen':[10, 25],
              'lstm__layers':[layers_hyper[0]],
              'lstm__optimizer_name':['adam']
              }
searcher_settings = {'scoring':'f1',
                     'n_iter':25,
                     'verbose': True}
fit_settings = {'lstm__epochs':150, 'lstm__shuffle':False}
cv_name = 'TimeSeriesSplit'
cv_settings = {'n_splits': 3}
_hypertuning1 = ml4qf.predictors.model_tuning.HyperTuning(pipe, searcher_name, searcher_settings,
                                                          hyper_grid, cv_name, cv_settings)
hypertuning1 = _hypertuning1()
hyperspace = list(product_dict(**hyper_grid))
score = []
for hi in hyperspace:
  tensorflow.keras.backend.clear_session()
  pipe.set_params(**hi)
  score_hi = []
  for cvi in hypertuning1.cv.split(Xtrain_reduced):
    index_train, index_test = cvi
    Xtrain_i = Xtrain_reduced[index_train]
    ytrain_i = y_train[index_train]
    Xtest_i = Xtrain_reduced[index_test]
    pipe.fit(Xtrain_i, ytrain_i, **fit_settings)
    ypred = pipe.predict(Xtest_i)
    score_i = sklearn.metrics.f1_score(y_train[index_test][SEQ_LEN-1:], ypred)
    score_hi.append(score_i)
  score.append(np.average(score_hi))
