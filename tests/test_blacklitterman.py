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
        pf1 = bl.BlackLitterman(1., Sigma, w_mkt, lambda_mkt)
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
