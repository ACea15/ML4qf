import numpy as np

class BlackLitterman:

    def __init__(self, lambda_portfolio, Sigma, Pi):

        self.lambda_portfolio = lambda_portfolio
        self.Sigma = Sigma
        self.Pi = Pi
        self.portfolio_settings = dict()

    def compute_Sigma_inv(self):
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        
    def compute_w_mkt(self):
        self.w_mkt = 1./self.lambda_portfolio * self.Sigma_inv * self.Pi

    def set_portfolio_settings(self, Omega=None, tau=None, P=None, Q=None):

        if Omega is not None:
            self._set_Omega(Omega)
        if tau is not None:
            self._set_tau(tau)
        if P is not None:
            self._set_P(P)
        if Q is not None:
            self._set_Q(Q)
        self.update_posterior = True
        assert self.tau is not None, "tau needs to not be None"
        assert self.P is not None, "P needs to not be None"
        assert self.Q is not None, "Q needs to not be None"
        if Omega is None:
            self._set_Omega(self.tau * self.P @ self.Sigma @ self.P.T) # dafault
            
    def _set_Omega(self, Omega):
        self.portfolio_settings['Omega'] = Omega
        self._Omega = Omega
        self.Omega_inv = np.linalg.inv(Omega)
    
    def _set_tau(self, tau):
        self.portfolio_settings['tau'] = tau
        self.tau = tau
    
    def _set_P(self, P):
        self.portfolio_settings['P'] = P
        self.P = P
    
    def _set_Q(self, Q):
        self.portfolio_settings['Q'] = Q
        self.Q = Q
        
    def compute_BLposterior(self):
        tauSigma_inv = 1./self.tau * self.Sigma_inv
        Pt = self.P.T
        self.Sigma_bl = np.linalg.inv(Pt @ self.Omega_inv @ self.P
                                      + tauSigma_inv)
        self.mu_bl = self.Sigma_bl @ (Pt @ self.Omega_inv @ self.Q
                                      + tauSigma_inv @ self.Pi)
        self.update_posterior = False
        
    def compute_w_bl(self):

        if self.update_posterior:
            self.compute_BLposterior()
        self.w_bl = (1./self.lambda_portfolio *
                     np.linalg.inv(self.Sigma_inv +
                                   self.Sigma_bl) @ self.mu_bl)
    

if __name__ == "__main__":
    ...
