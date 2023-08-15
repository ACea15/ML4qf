import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
def regression_OLS(X, y):
    model = sm.OLS(y, X)
    model = model.fit()
    return model

def arima_pred(x_train, num_steps, arima_parameters, fit_sett=None):
    if fit_sett is None:
        fit_sett = dict()
    history = list(x_train)
    predictions = list()
    # walk-forward validation
    for t in range(num_steps):
        model = ARIMA(history, order=arima_parameters)
        model_fit = model.fit(**fit_sett)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append(yhat)
    return np.array(predictions)

def arima_fit(X, model_names, arima_parameters,
              model_sett=None, fit_sett=None):
    if fit_sett is None:
        fit_sett = dict()
    if model_sett is None:
        model_sett = dict()
    arima_models = dict()
    if isinstance(arima_parameters, dict):
        parameters_asdict = True
    else:
        parameters_asdict = False
    for i, k in enumerate(model_names):
        print(f"Fitting ARIMA for {k}")
        if parameters_asdict:
            model_k = ARIMA(X[:, i], order=arima_parameters[k],
                            **model_sett)
        else:
            model_k = ARIMA(X[:, i], order=arima_parameters,
                            **model_sett)
        model_fit = model_k.fit(**fit_sett)
        arima_models[k] = model_fit
    return arima_models

def arima_build_pred(arima_model, Xtrain, Xtest,
                     model_names, index_train = None, index_test = None):

    train_dict = dict()
    test_dict = dict()
    len_train = len(Xtrain)
    len_test = len(Xtest)    
    for i, k in enumerate(model_names):
        train_dict[k] = Xtrain[:, i]
        train_dict[k+"_pred"] = arima_model[k].predict(0, len_train-1)
        test_dict[k] = Xtest[:, i]
        test_dict[k+"_pred"] = arima_model[k].predict(len_train, len_train + len_test -1)
    df_train = pd.DataFrame(train_dict, index=index_train)
    df_test = pd.DataFrame(test_dict, index=index_test)
    return df_train, df_test

def err_rmse(y, yhat):
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return rmse


# model_fit = model.fit()
# # summary of fit model
# print(model_fit.summary())
# # line plot of residuals
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# # density plot of residuals
# residuals.plot(kind='kde')
# pyplot.show()
