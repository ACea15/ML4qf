import statsmodels.api as sm

def regression_OLS(X, y):
    model = sm.OLS(y, X)
    model.fit()
    return model
