import numpy as np
import os
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import models_dict

class ModelPrediction:

    """
    Build predictor model
    """
    
    def __init__(self, predictor_name, **kwargs):

        self.settings = kwargs
        self.predictor_name = predictor_name
        self.predictor = models_dict[predictor_name](**self.settings)

    def run(self, X_train=None, y_train=None, X_test=None, y_test=None,
            print_info=True, **kwargs):

        # fit the predictor model
        if X_train is not None and y_train is not None:
            self.Xtrain = X_train
            self.ytrain = y_train
            self.predictor.fit(X_train, y_train)
            self.ytrain_pred = self.predictor.predict(X_train)
        if X_test is not None and y_test is not None:
            self.Xtest = X_test
            self.ytest = y_test
            self.ytest_pred = self.predictor.predict(X_test)

        # Classification Report
        if print_info:

            self.predictor_info()
            
    def set_signal(self, label, df_train=None, df_test=None):

        return_df = []
        if df_train is not None:
            df_train[''.join(['signal_', label])] = self.ytrain_pred
            return_df.append(df_train)
        if df_test is not None:
            df_test[''.join(['signal_', label])] = self.ytest_pred
            return_df.append(df_test)
        if len(return_df) == 1:
            return return_df[0]
        elif len(return_df) == 2:
            return return_df
        else:
            raise ValueError("No df provided")
        
    def predictor_info(self):

        if self.Xtrain is not None and self.ytrain is not None:
            print('#####################################################')
            print('####### Classification_report (Training data) #######')
            print('#####################################################')
            print(classification_report(self.ytrain, self.ytrain_pred))
            print('#####################################################')
        if self.Xtrain is not None and self.ytrain is not None:
            print('####### Classification_report (Testing data)  #######')
            print('#####################################################')
            print(classification_report(self.ytest, self.ytest_pred))
            print('#####################################################')

class ModelHyperTuning:


    def __init__(self, predictor_name, searcher_name, hyper_grid,
                predictor_settings, searcher_settings, cv_settings):

        self.model = ModelPrediction(predictor_name, **predictor_settings)
        self.tscv = split_timedata(X=None, y=None, output_splitdata=False, **cv_settings)
        self.searcher_type = getattr(model_selection, searcher_name)

        self.searcher = self.searcher_type(self.model.predictor,
                                           hyper_grid,
                                           cv=self.tscv,
                                           **searcher_settings)
        self.predictor_name = predictor_name
        self.searcher_name = searcher_name
        self.hyper_grid = hyper_grid
        self.searcher_settings = searcher_settings
        self.predictor_settings = predictor_settings

    @property
    def model_params(self):

        # Get params list
        return self.model.predictor.get_params()

    def run(self, X_train, y_train, print_info=True, **kwargs):

        self.searcher.fit(X_train, y_train, **kwargs)

        if print_info:
            # best parameters
            print(self.searcher.best_params_)
            # best score
            print(self.searcher.best_score_)

    def new(self, X_train=None, y_train=None, X_test=None, y_test=None,
            print_info=True, **kwargs):
        # Refit the XGB Classifier with the best params
        self.newmodel = ModelPrediction(self.predictor_name, **self.searcher.best_params_)
        self.newmodel.run(X_train, y_train, X_test, y_test, print_info, **kwargs)


# # Create a model
# def create_model(hu=256, lookback=60):

#     tensorflow.keras.backend.clear_session()   
    
#     # instantiate the model
#     model = Sequential()
#     model.add(LSTM(units=hu, input_shape=(lookback, 1), activation = 'relu', return_sequences=False, name='LSTM'))
#     model.add(Dense(units=1, name='Output'))              # can also specify linear activation function 
    
#     # specify optimizer separately (preferred method))
# #     opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#     opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       # adam optimizer seems to perform better for a single lstm
    
#     # model compilation
#     model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
#     return model


# # Specify callback functions
# model_path = (results_path / 'model.h5').as_posix()
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# my_callbacks = [
#     EarlyStopping(patience=10, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
#     ModelCheckpoint(filepath=model_path, verbose=1, monitor='loss', save_best_only=True),
#     TensorBoard(log_dir=logdir, histogram_freq=1)
# ]



# # Model fitting
# lstm_training = model.fit(X_train, 
#                           y_train, 
#                           batch_size=64, 
#                           epochs=500, 
#                           verbose=1, 
#                           callbacks=my_callbacks, 
#                           shuffle=False)


lstm_type
optimizer
optimizer_settings
loss_function
metrics
layers

# compile model five
nn = Sequential()
nn.add(LSTM(50, activation='relu', input_shape=(3, 1)))
nn.add(Dense(1))
nn.compile(optimizer='adam', loss='mse')
print(nn.summary())

# fit model
nn.fit(X, y, batch_size=5, epochs=2000, validation_split=0.2, verbose=0)


### Inputs...

lookback = 60
layers = dict()
layers.update(LSTM={'units':4,
                    'input_shape':(lookback, 1),
                    'activation':'relu',
                    'return_sequences': False,
                    'name': 'LSTM'})

lstm_type = 'Sequential'
OPTIMIZER = 'Adam' # RMSprop
OPTIMIZER_SETT = dict()
OPTIMIZER_SETT.update(lr=0.001, epsilon=1e-08, decay=0.0)
COMPILATION_SETT = {'loss':'mse', 'metrics':['mae']}
# tensorflow modules
import tensorflow.keras.models as tf_models #Sequential
import tensorflow.keras.layers as tf_layers #import Dense, Dropout, Flatten, LSTM
import tensorflow.keras.optimizers as tf_optimizers #import Adam, RMSprop 
from tensorflow.keras.utils import plot_model
import tensorflow.keras.callbacks as tf_callbacks#import EarlyStopping, ModelCheckpoint, TensorBoard


# build model
model = getattr(tf_models, lstm_type)
for k, v in layers.items():
    layer_i = getattr(tf_layers, k)
    model.add(layer_i(**v))

# build optimizer
optimizer = getattr(tf_optimizers, optimizer)
opt = optimizer(**OPTIMIZER_SETT)

# train the model
model.compile(optimizer=opt, **COMPILATION_SETT)

class KerasModel:

    def __init__(self):
        pass

class SKModel:

    def __init__(self):
        pass
    

# Model fitting
lstm_training = model.fit(X_train, 
                          y_train, 
                          batch_size=64, 
                          epochs=500, 
                          verbose=1, 
                          callbacks=my_callbacks, 
                          shuffle=False)



# #     opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#     opt = Adam(lr=0.001, epsilon=1e-08, decay=0.0)       # adam optimizer seems to perform better for a single lstm


# predict the outcome
test_input = np.array([70,71,72])
test_input = test_input.reshape((1, 3, 1))
test_output = nn.predict(test_input, verbose=0)
print(test_output)


### Inputs...

lookback = 60
layers = dict()
layers.update(LSTM={'units':4,
                    'input_shape':(lookback, 1),
                    'activation':'relu',
                    'return_sequences': False,
                    'name': 'LSTM'})


TICKER ='BSP.F'
YEAR0 = 2022
MONTH0 = 1
DAY0 = 15
YEARS_MODELLING = 5.5
YEARS_BACKTESTING = 0
YEAR = YEAR0 - YEARS_BACKTESTING
NUM_DAYS = 365 * YEARS_MODELLING 
df0 = f1.get_data(TICKER, YEAR, MONTH0, DAY0, NUM_DAYS)

FEATURES1 = {'return_': [1, 2, 5, 8, 15, 23],
            'momentum_': [1, 2, 5, 8, 15, 23],
            'OC_': None,
            'HL_': None,
            'Ret_': list(range(10, 90, 10)),
            'Std_': list(range(10, 90, 10)),
            'MA_': [5, 10, 25, 50],
            'EMA_': [5, 10, 25, 50],
            'sign_return_': [1, 2, 5, 8, 15, 23],
            'sign_momentum_':[1, 2, 5, 8, 15, 23]
            }
df_x1, features_m1 = f1.get_features(df0, price='Adj Close', 
                                     log_return=True, **FEATURES1)
