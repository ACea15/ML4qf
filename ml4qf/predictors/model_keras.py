import numpy as np
import pickle
from sklearn.base import BaseEstimator

# tensorflow modules
import tensorflow.keras.models as tf_models  # Sequential
import tensorflow.keras.layers as tf_layers  # import Dense, Dropout, Flatten, LSTM
import tensorflow.keras.optimizers as tf_optimizers  # import Adam, RMSprop
import tensorflow.keras.losses as tf_losses
import tensorflow.keras.preprocessing.sequence as tf_sequence
from tensorflow.keras.utils import plot_model
import tensorflow.keras.callbacks as tf_callbacks # import EarlyStopping,
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf

def to_dict(*args):

    out = list()
    for i, args_i in enumerate(args):
        if args_i == None:
            out.append(dict())
        else:
            out.append(args_i)
    return out

class Model(BaseEstimator):

    def __init__(self,  keras_model='Sequential', layers=(), seqlen=0,
                 optimizer_name='adam', loss_name='mse', metrics=None,
                 optimizer_sett=None, compile_sett=None, loss_sett=None,
                 timeseries_sett=None):
        """

        Parameters
        ----------
        fit_sett :

        optimizer_sett :

        layers :

        keras_model :

        optimizer_name :


        Returns
        -------
        out :

        """

        self.keras_model = keras_model
        self.layers = layers
        self.seqlen = seqlen
        self.loss_name = loss_name
        self.loss_sett = loss_sett        
        self.metrics = metrics
        self.optimizer_name = optimizer_name
        self.optimizer_sett = optimizer_sett
        self.compile_sett = compile_sett
        self.timeseries_sett = timeseries_sett
        
    def fit(self, X, y, **kwargs):
        """Fit method

        Parameters
        ----------
        X : np.array
            Input parameters
        y : labels
            data to learn from
        **kwargs : dict

        """
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]        
        self.build()
        self.set_callbacks()
        if self.seqlen > 0:
            self.X_generated = tf_sequence.TimeseriesGenerator(X,
                                                               y,
                                                               length=self.seqlen)
            self._fit_history = self._model.fit(self.X_generated,
                                                callbacks=self._callbacks,
                                                **kwargs
                                                )
        else:
            self._fit_history = self._model.fit(X,
                                                y,
                                                callbacks=self._callbacks,
                                                **kwargs
                                                )
        return self

    # def predict(self, X: np.array, **kwargs):

    #     return self.model.predict(X, **kwargs)

    def get_params(self, deep=True):

        dic = {"keras_model": self.keras_model,
               "layers": self.layers,
               "seqlen": self.seqlen,
               "optimizer_name": self.optimizer_name,
               "optimizer_sett": self.optimizer_sett,
               "compile_sett": self.compile_sett,
               "loss_sett": self.loss_sett,
               "loss_name": self.loss_name,
               "metrics": self.metrics,
               "timeseries_sett": self.timeseries_sett
               }
        return dic

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def build_sequential(self):

        self._model = tf_models.Sequential()
        if len(self.layers) == 0: #dafault model
            # self._model.add(LSTM(units=5, activation = 'relu', return_sequences=False, name='LSTM'))
            # self._model.add(Dense(units=1, name='Output'))
            self._model.add(tf.keras.layers.Dense(8))
            # Afterwards, we do automatic shape inference:
            self._model.add(tf.keras.layers.Dense(4))

        else:
            for i, li in enumerate(self.layers):
                li_name = li[0]
                li_dict = {x[0]:x[1] for x in li[1]}
                if i == 0:
                    li_dict['input_shape'] = (self.seqlen, self.n_features_in_)
                layer_i = getattr(tf_layers, li_name)
                self._model.add(layer_i(**li_dict))

    def build_model(self):

        self._model = tf_models.Model()

    def build(self):
        """


        """

        if self.keras_model == 'Sequential':
            self.build_sequential()
        elif self.keras_model == 'Model':
            self.build_model()
        self.set_optimizer()
        self.set_loss()
        if self.compile_sett is not None:
            self._model.compile(optimizer=self._optimizer,
                                loss=self._loss,
                                metrics=self.metrics,
                                **self.compile_sett
                                )
        else:
            self._model.compile(optimizer=self._optimizer,
                                loss=self._loss,
                                metrics=self.metrics,
                                )

    def set_optimizer(self):

        try:
            optimizer = getattr(tf_optimizers, self.optimizer_name)
            if self.optimizer_sett is not None:
                self._optimizer = optimizer(**self.optimizer_sett)
            else:
                self._optimizer = optimizer()
        except AttributeError:
            self._optimizer = self.optimizer_name

    def set_loss(self):

        try:
            raise AttributeError
            loss = getattr(tf_losses, self.loss_name)
            if type(loss).__name__ == 'function':
                self._loss = loss
            else:
                if self.loss_sett is not None:
                    self._loss = loss(**self.loss_sett)
                else:
                    self._loss = loss()
        except AttributeError:
            self._loss = self.loss_name

    def set_callbacks(self):

        self._callbacks = None


class Model_binary(Model):

    def __init__(self,  keras_model='Sequential', layers=(), seqlen=0,
                 optimizer_name='adam', loss_name='binary_crossentropy', metrics=None,
                 optimizer_sett=None, compile_sett=None, loss_sett=None, timeseries_sett=None):

        super().__init__(keras_model, layers, seqlen, optimizer_name,
                         loss_name, metrics, optimizer_sett, compile_sett,
                         loss_sett, timeseries_sett)
        
    def predict(self, X, y):
        X_generated = tf_sequence.TimeseriesGenerator(X,
                                                      y,
                                                      length=self.seqlen)
        y = self._model.predict(X_generated)
        ypred = np.round(y)[:, 0]
        return ypred

if (__name__ == '__main__'):


    from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(LinearSVC())  # passes

    check_estimator(Model())


    # # summarize the sonar dataset
    # from pandas import read_csv
    # # load dataset
    # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
    # dataframe = read_csv(url, header=None)
    # # split into input and output elements
    # data = dataframe.values
    # X, y = data[:, :-1], data[:, -1]
    # print(X.shape, y.shape)
