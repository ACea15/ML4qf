__all__ = ["factor_lin_model",
           "get_factors",
           "get_factor_names",
           "factors_regression",
           "compute_factors_coeff",
           "factor_lin_generator"]

import pandas as pd
import statsmodels.api as sm
import getFamaFrenchFactors as gff
from datetime import date
from functools import partial


def get_factors(factor_names: list[str], frequency: str):
    df = None
    for fi in factor_names:
        obj_fi = getattr(gff, fi)
        df_fi = obj_fi(frequency=frequency)
        df_fi.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
        df_fi.set_index('Date', inplace=True)
        if df is None:
            df = df_fi
        else:
            df = df.join(df_fi)
    return df

def get_factor_names(factors):
    
    factor_names = []
    for k, v in factors.items():
        if isinstance(v, str):
            factor_names.append(v)
        elif isinstance(v, list):
            factor_names += v

def factors_regression(factor_names, df_gff, df_assets, regression_kernel):

    df_factors = df_gff[factor_names]
    X = sm.add_constant(df_factors)
    models = dict()
    for k, rets in df_assets.items():
        y = rets - df_gff['RF']
        model_k = regression_kernel(X, y)
        models[k] = model_k
    return models

def compute_factors_coeff(models):

    alpha = list()
    beta = list()
    for asset_k in models.keys():
        alpha_k, *beta_k = models[asset_k].params
        alpha.append(alpha_k)
        beta.append(beta_k)
    alpha = np.array(alpha).reshape((1, len(alpha)))
    beta = np.array(beta).T

    return alpha, beta

def factor_lin_model(X, alpha, beta):

    return X @ beta + alpha

def factor_lin_generator(alpha, beta):

    return partial(factor_lin_model, alpha=alpha, beta=beta)
