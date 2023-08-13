import numpy as np
import ml4qf.portfolios.blacklitterman as bl
import pytest

class TestBL_Seb:

    @pytest.fixture(scope="class")
    def inputs(self):

        sigma_mkt = 0.2235
        sharpe_ratio = 0.5
        lambda_mkt = sharpe_ratio / sigma_mkt
        Sigma = np.array([[0.0049, 0.00672, 0.0105, 0.0168],
                          [0.00672, 0.0144, 0.0252, 0.036],
                          [0.0105, 0.0252, 0.09, 0.144],
                          [0.0168, 0.036, 0.144, 0.36]
                          ])
        w_mkt = np.array([0.05, 0.4, 0.45, 0.1])
        tau = 0.833e-2
        P = np.array([[-1, 0, 1, 0],
                      [0, 1, 0, 0]])
        Q = np.array([0.1, 0.03])
        return (lambda_mkt, w_mkt, Sigma, tau, P, Q)

    @pytest.fixture(scope="class")
    def init_obj(self, inputs):

        (lambda_mkt, w_mkt, Sigma, tau, P, Q) = inputs
        pf1 = bl.BlackLitterman(Sigma, w_mkt, lambda_mkt)
        pf1.set_portfolio_inputs(tau, P, Q)
        return pf1

    @pytest.fixture
    def solution(self):

        mu_mkt = np.array([0.02089038, 0.0470604 , 0.14654362, 0.25959732])        
        mu_bl = np.array([0.01677028, 0.03753062, 0.12476476, 0.22701031])
        Sigma_bl = np.array([[2.76539063e-05, 2.72595454e-05, 3.34863647e-05, 6.17560471e-05],
                             [2.72595454e-05, 5.47443909e-05, 6.91010946e-05, 9.09994686e-05],
                             [3.34863647e-05, 6.91010946e-05, 3.20263996e-04, 5.33152721e-04],
                             [6.17560471e-05, 9.09994686e-05, 5.33152721e-04, 1.95991219e-03]])

        return mu_mkt, mu_bl, Sigma_bl

    def test_BLmu_mkt(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.mu_mkt, mu_mkt) 

    def test_BLmu(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.mu_bl, mu_bl) 

    def test_BLSigma(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.Sigma_bl, Sigma_bl)

class TestBL_Stauton:

    @pytest.fixture(scope="class")
    def inputs(self):

        w_mkt = np.array([19.34e-2,
                         26.13e-2,
                         12.09e-2,
                         12.09e-2,
                         1.34e-2,
                         1.34e-2,
                         24.18e-2,
                         3.49e-2
                         ])

        Sigma = np.array([[0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437],
                          [0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535],
                          [-0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039],
                          [-0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854],
                          [0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994],
                          [0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235],
                          [-0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064],
                          [-0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.079958]])

        risk_prem = 0.03
        lambda_mkt = risk_prem / w_mkt.T.dot(Sigma.dot(w_mkt))

        Q = np.array([0.0525, 0.0025, 0.02])
        P = np.array([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
                      [-1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.90, -0.90, 0.10, -0.10, 0.00, 0.00]])
        tau = 0.025
        return (lambda_mkt, w_mkt, Sigma, tau, P, Q)

    @pytest.fixture(scope="class")
    def init_obj(self, inputs):

        (lambda_mkt, w_mkt, Sigma, tau, P, Q) = inputs
        pf1 = bl.BlackLitterman(Sigma, w_mkt, lambda_mkt)
        pf1.set_portfolio_inputs(tau, P, Q)
        return pf1

    @pytest.fixture
    def solution(self):

        mu_mkt = np.array([0.08, 0.67, 6.42, 4.08, 7.43, 3.7, 4.8, 6.6])*1e-2
        Sigma_bl = np.linalg.inv([[66439.14, -19665.92, -1570.25, 2014.38,
                                   -233.19, -3220.85, 5087.40, -465.52],
                                  [-19665.92, 15956.06, 787.02,   85.67,
                                   348.11,   823.42,  -3118.29, 344.56],
                                  [-1570.25,  787.02,   6765.86, -5503.89,
                                   -3894.49, 4966.53, -1682.60, 203.28],
                                  [2014.38,   85.67,    -5503.89, 8918.27,
                                   4289.63, -6309.41, -1349.44, -672.24],
                                  [-233.19,   348.11,   -3894.49, 4289.63,
                                   4466.67, -5017.04, -530.77, -659.17],
                                  [-3220.85,  823.42,   4966.53, -6309.41,
                                   -5017.04, 8839.05, -752.29,  248.89],
                                  [5087.40,   -3118.29, -1682.60,-1349.44,
                                   -530.77, -752.29,  8216.83, -760.64],
                                  [-465.52,   344.56,   203.28,  -672.24,
                                   -659.17,  248.89,  -760.64,  1410.47]])

        mu_bl = np.array([0.07e-2,
                          0.50e-2,
                          6.50e-2,
                          4.32e-2,
                          7.59e-2,
                          3.94e-2,
                          4.93e-2,
                          6.84e-2])

        return mu_mkt, mu_bl, Sigma_bl

    def test_BLmu_mkt(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.mu_mkt, mu_mkt, 10e-6, 10e-5) 

    def test_BLmu(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.mu_bl, mu_bl, 10e-6, 10e-5) 

    def test_BLSigma(self, init_obj, solution):
        
        mu_mkt, mu_bl, Sigma_bl = solution
        assert np.allclose(init_obj.Sigma_bl, Sigma_bl, 10e-6, 10e-6)
