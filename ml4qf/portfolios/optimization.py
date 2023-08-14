import numpy as np
import scipy.optimize

def mean_variance_opt(mu, Sigma_inv, lambda_portfolio):
    w_opt = 1. / lambda_portfolio * Sigma_inv @ mu
    return w_opt

class MinimizeFuns:

    @staticmethod
    def mean_variance(x, *args):

        mu, Sigma, *xargs = args
        lambda_portfolio = xargs[0]
        portfolio_mean_return = np.dot(mu, x)
        portfolio_variance = np.dot(x, Sigma @ x)
        return (-portfolio_mean_return + 0.5 *
                lambda_portfolio * portfolio_variance)

    @staticmethod
    def variance(x, *args):

        mu, Sigma, *xargs = args
        portfolio_variance = np.dot(x, Sigma @ x)
        return portfolio_variance

    @staticmethod
    def sharpe(x, *args):

        mu, Sigma, risk_free, *xargs = args
        portfolio_mean_return = np.dot(mu, x)        
        portfolio_variance = np.dot(x, Sigma @ x)
        sharpe_ratio = ((portfolio_mean_return - risk_free) /
                        np.sqrt(portfolio_variance))
        return sharpe_ratio

class ConstraintsFuns:

    @staticmethod
    def eq_weights1(x, *args):
        return np.sum(x) - 1

    @staticmethod
    def eq_rets(x, *args):
        mu, Sigma, *xargs = args
        target_rets = xargs[0]
        portfolio_mean_return = np.dot(mu, x)
        return portfolio_mean_return - target_rets

    @staticmethod
    def ieq_weights0(x, *args):
        return x

def scipy_minimize(fun_name: str,
                   x0: np.ndarray,
                   method_name: str,
                   args: tuple=(),
                   cons_sett: dict=None,
                   **kwargs):

    fun = getattr(MinimizeFuns, fun_name)
    if cons_sett is not None:
        constraints = list()
        for k, v in cons_sett.items():
            c_fun = getattr(ConstraintsFuns, k)
            c_type = v.get('type', 'eq')
            c_jac = v.get('jac', None)
            c_dict = dict(fun=c_fun,
                          type=c_type,
                          jac=c_jac,
                          args=args)
            constraints.append(c_dict)
    else:
        constraints = ()
            
    res = scipy.optimize.minimize(fun, x0, args,
                                  method=method_name,
                                  constraints=constraints,
                                  **kwargs)
    return res

    

    
