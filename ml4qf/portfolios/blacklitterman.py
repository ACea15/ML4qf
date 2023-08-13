import numpy as np

class BlackLitterman:

    def __init__(self, Sigma, w_mkt, lambda_mkt):

        self.Sigma = Sigma
        self.w_mkt = w_mkt
        self.portfolio_settings = dict()
        self.lambda_mkt = lambda_mkt
        self.compute_Sigma_inv()
        
    def compute_Sigma_inv(self):
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        
    def compute_mu_mkt(self):
        self.mu_mkt = self.lambda_mkt * self.Sigma @ self.w_mkt

    def set_portfolio_inputs(self, tau=None, P=None, Q=None, Omega=None):

        if Omega is not None:
            self._set_Omega(Omega)
        if tau is not None:
            self._set_tau(tau)
        if P is not None:
            self._set_P(P)
        if Q is not None:
            self._set_Q(Q)
        assert self.tau is not None, "tau needs to not be None"
        assert self.P is not None, "P needs to not be None"
        assert self.Q is not None, "Q needs to not be None"
        if Omega is None:
            Omega = np.diag(np.diagonal(self.tau * self.P @ self.Sigma @ self.P.T))
            self._set_Omega(Omega) # dafault
        self.compute_mu_mkt()
        self.compute_BLposterior()
        
    def _set_Omega(self, Omega):
        self.portfolio_settings['Omega'] = Omega
        self.Omega = Omega
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
                                      + tauSigma_inv @ self.mu_mkt)
        
    

if __name__ == "__main__":
    ...
