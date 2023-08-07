import numpy as np

Sigma = np.array([[0.0049, 0.00672, 0.0105, 0.0168],
                  [0.00672, 0.0144, 0.0252, 0.036],
                  [0.0105, 0.0252, 0.09, 0.144],
                  [0.0168, 0.036, 0.144, 0.36]
                  ])
w_mkt = np.array([0.05, 0.4, 0.45, 0.1])
risk_aver_portfolio = np.array([0.1, 2.24, 6])

sigma_mkt = 0.2235
sharpe_ratio = 0.5
lambda_portfolio = sigma_mkt * sharpe_ratio

tau = 0.833e-2

P = np.array([[-1, 0, 1, 0],
              [0, 1, 0, 0]])
Q = np.array([0.1, 0.03])

mu_bl = np.array([1.68e-2, 3.75e-2, 12.48e-2, 22.70e-2])

Sigma_bl = np.array([[0.0000276650,	0.0000272705,	0.00003,	0.00006],
                     [0.0000272705,	0.000054766,	0.00007,	0.00009],
                     [0.00003,	        0.00007,	0.00032,	0.00053],
                     [0.00006,	        0.00009,	0.00053,	0.00196]])

