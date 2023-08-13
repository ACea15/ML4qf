import numpy as np

def mean_variance_opt(mu, Sigma_inv, lambda_portfolio):
    w_opt = 1. / lambda_portfolio * Sigma_inv @ mu
    return w_opt

def mean_variance(x, *args):
    
    mu, Sigma = args
    portfolio_mean_return = np.dot(mu, x)
    portfolio_variance = np.dot(x, Sigma @ x)
    return portfolio_mean_return + 0.5 * portfolio_variance
