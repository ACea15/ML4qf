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
from sklearn.model_selection import train_test_split
import sklearn.metrics
import scipy.optimize
import sklearn.compose

import plotly.express as px
import scipy.stats
import pandas as pd
import pandas_ta as ta
import ml4qf.inputs
import ml4qf.collectors.financial_features
import ml4qf.transformers
import ml4qf.predictors.model_som


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

########
# def set_seeds(seed=42): 
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

ml4qf.utils.set_seeds(['np.random'])
som_size = 50
som_num_features = Xtrain_scaled.shape[1]
som_model = minisom.MiniSom(som_size, som_size, som_num_features, sigma=1.5, learning_rate=0.1, 
neighborhood_function='gaussian', random_seed=42)
# x, y, input_len, sigma=1.0, learning_rate=0.5,
#                  decay_function=asymptotic_decay,
#                  neighborhood_function='gaussian', topology='rectangular',
#                  activation_distance='euclidean', random_seed=None)
som_model.pca_weights_init(Xtrain_scaled)
som_model.train(Xtrain_scaled, 10000, verbose=True)

W = som_model.get_weights()
som_labels0, target_name = ml4qf.predictors.model_som.Model.feature_selection(W, labels=df_.columns, target_index = -1, a = 0.08)

#assert target_name = 'target', "targets do not coincide after som" 
#dftrain_reduced = df_train[som_labels]
#dftest_reduced = df_test[som_labels]
som_labels0

import ml4qf.predictors.model_som
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
#+begin_src python :session py1 :results file
fig1_path= img_dir +'/som.png'
fig1 = px.imshow(W[:,:,0].T)
fig1.layout.height = 600
fig1.layout.width = 600
fig1.write_image(fig1_path)
fig1_path

index_reducedlabels = [df_train.columns.get_loc(i) for i in som_labels]
dftrain_reduced = df_train[som_labels]
dftest_reduced = df_test[som_labels]
assert (dftrain_reduced.to_numpy() == Xtrain[:, index_reducedlabels]).all(), "Reduced matrix not maching dimensions"
Xtrain_reduced = Xtrain_scaled[:, index_reducedlabels]
Xtest_reduced = Xtest_scaled[:, index_reducedlabels]
#Xtest_reduced = Xtest_scaled[:, index_reducedlabels]

import ml4qf.predictors.model_keras as model_keras
SEQ_LEN = 10
y = df_train['target'].to_numpy()
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
layers_dict['LSTM_1'] = dict(units=100, activation = 'elu', return_sequences=True, name='LSTM1')
layers_dict['LSTM_2'] = dict(units=100, activation = 'elu', return_sequences=False, name='LSTM2')
layers_dict['Dense_1'] = dict(units=1, activation='sigmoid', name='Output')
#####################
layers_tuple = ml4qf.utils.dict2tuple(layers_dict)
#######################
base_model = model_keras.Model_binary(keras_model='Sequential', layers=layers_tuple,
                                      seqlen=SEQ_LEN, optimizer_name='adam',
                                      loss_name='binary_crossentropy', metrics=['accuracy','binary_accuracy', 'mse'],
                                      optimizer_sett=None, compile_sett=None, loss_sett=None)
base_model.fit(Xtrain_reduced, y, epochs=50, shuffle=False, verbose=1)


y_test  = df_test.target.to_numpy()
ypred_basemodel = base_model.predict(Xtest_reduced, y_test)
print(sklearn.metrics.classification_report(y_test[:-9], ypred_basemodel.reshape(len(ypred_basemodel))))

ypred_basemodeltrain = base_model.predict(Xtrain_reduced, y)
print(sklearn.metrics.classification_report(y[:-9], ypred_basemodeltrain.reshape(len(ypred_basemodeltrain))))

# summary
base_model._model.summary()

# base_model._model.compute_loss(y_true, y_pred)


import ml4qf.predictors.model_keras
Xt1, yt1 = ml4qf.predictors.model_keras.Model.split_data(Xtest_reduced,SEQ_LEN ,y_test)
# Test the model after training
test_results = base_model._model.evaluate(Xt1, yt1, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

Xtr1, ytr1 = ml4qf.predictors.model_keras.Model.split_data(Xtrain_reduced,SEQ_LEN ,y)
# Test the model after training
test_results = base_model._model.evaluate(Xtr1, ytr1, verbose=1, batch_size=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

import tensorflow.keras.metrics as tf_metrics
m2 = tf_metrics.Accuracy()
m2.update_state(base_model.y_predicted_, ytr1)
m2.result().numpy()


m = tf_metrics.Accuracy()
m.update_state([[1], [2], [3], [4]], [[0], [2], [3], [4]])
m.result().numpy()
# import tensorflow.keras.losses as tf_losses
# bce = tf_losses.BinaryCrossentropy()

# bce(y_test, base_model.y_predicted_.reshape(len(base_model.y_predicted_)))

# y_true = [0, 1, 0, 0]
# y_pred = [0,1,0,1]
# bce = tf_losses.BinaryCrossentropy(from_logits=True)
# bce(y_true, y_pred).numpy()
# from numpy import array
# from numpy import hstack
# # define input sequence
# in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))
# print(dataset)

# # split a multivariate sequence into samples
# def split_sequences(sequences, n_steps):
# 	X, y = list(), list()
# 	for i in range(len(sequences)):
# 		# find the end of this pattern
# 		end_ix = i + n_steps
# 		# check if we are beyond the dataset
# 		if end_ix > len(sequences):
# 			break
# 		# gather input and output parts of the pattern
# 		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
# 		X.append(seq_x)
# 		y.append(seq_y)
# 	return array(X), array(y)


# def split_data(X_in, n_steps, y_in=None):

#     X, y = list(), list()
#     for i in range(len(X_in)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the dataset
#         if end_ix > len(X_in):
#                 break
#         # gather input and output parts of the pattern
#         seq_x = X_in[i:end_ix, :]
#         seq_y = y_in[end_ix-1]
#         X.append(seq_x)
#         y.append(seq_y)
#     if y_in is None:
#         return np.array(X)
#     else:
#         return np.array(X), np.array(y)

# # define input sequence
# in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# # convert to [rows, columns] structure
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# # horizontally stack columns
# dataset = hstack((in_seq1, in_seq2, out_seq))
# # choose a number of time steps
# n_steps = 3
# # convert into input/output
# X, y = split_sequences(dataset, n_steps)
# print(X.shape, y.shape)
# # summarize the data
# for i in range(len(X)):
# 	print(X[i], y[i])


# X2= np.vstack([[10, 20, 30, 40, 50, 60, 70, 80, 90],
#               [15, 25, 35, 45, 55, 65, 75, 85, 95]]).T
# y2 = np.array([ 25,  45,  65,  85, 105, 125, 145, 165, 185])
             
# import tensorflow.keras.preprocessing.sequence as tf_sequence
# X_generated = tf_sequence.TimeseriesGenerator(X2,
#                                               y2,
#                                               length=3)
# X3, y3 = split_data(X2, 3, y2)

