import statsmodels.api as sm
import numpy as np
import pytest
import ml4qf.predictors.model_stats as ms

class TestBL_Stauton:

    @pytest.fixture(scope="class")
    def inputs(self):

        data=np.array([[0.011,   0.143],   
                       [0.070,   0.128],   
                       [0.050,   0.055],   
                       [0.010,   0.007],   
                       [-0.017,  -0.061],
                       [0.040,   0.245],   
                       [-0.011,  0.014],   
                       [-0.156,  -0.136],
                       [0.062,   0.137],   
                       [0.078,   -0.039],
                       [0.059,   0.142],   
                       [0.056,   0.128],   
                       [0.041,   0.233],   
                       [-0.032,  -0.153],
                       [0.039,   0.177],   
                       [0.038,   -0.097],
                       [-0.024,  -0.008],
                       [0.054,   0.111],   
                       [-0.032,  -0.050],
                       [-0.005,  0.076],   
                       [-0.028,  -0.022],
                       [0.061,   0.022],   
                       [0.020,   -0.017],
                       [0.057,   0.249],   
                       [-0.052,  -0.176],
                       [-0.019,  -0.091],
                       [0.093,   0.173],   
                       [-0.031,  -0.421],
                       [-0.021,  -0.109],
                       [0.024,   0.246],   
                       [-0.016,  -0.136],
                       [0.060,   0.000],   
                       [-0.054,  -0.146],
                       [-0.004,  0.133],   
                       [-0.082,  -0.183],
                       [0.005,   -0.280],
                       [0.035,   0.342],   
                       [-0.096,  -0.034],
                       [-0.065,  -0.076],
                       [0.075,   0.214],   
                       [0.007,   0.021],   
                       [-0.025,  0.054],   
                       [-0.010,  -0.098],
                       [-0.065,  -0.149],
                       [-0.084,  -0.109],
                       [0.019,   0.128],   
                       [0.074,   0.099],   
                       [0.009,   0.031],   
                       [-0.015,  -0.039],
                       [-0.019,  -0.088],
                       [0.037,   0.033],   
                       [-0.063,  -0.143],
                       [-0.007,  -0.026],
                       [-0.074,  0.072],   
                       [-0.081,  -0.131],
                       [0.007,   0.023],   
                       [-0.115,  -0.115],
                       [0.084,   0.201],   
                       [0.057,   0.076],   
                       [-0.061,  -0.109]])

        return data

    @pytest.fixture(scope="class")
    def init_obj(self, inputs):

        data = inputs
        X = data[:, 0]
        X = sm.add_constant(X)
        y = data[:, 1]
        model = ms.regression_OLS(X, y)
        return model

    @pytest.fixture
    def solution(self):

        alpha = 0.0087489
        beta = 1.6854573

        return alpha, beta

    def test_coeff(self, init_obj, solution):
        
        alpha, beta  = solution
        alpha_calculated, beta_calculated = init_obj.params
        assert abs(alpha - alpha_calculated) < 1e-4
        assert abs(beta - beta_calculated) < 1e-4

if __name__ == "__main__":
    import ml4qf.collectors.financial_features as ff
    dates = ff.get_ticker_dates(1997, 12, 15, 2002, 12, 31)
    fdc = ff.FinancialDataContainer(["^GSPC", "MSFT"],
                                    dates[0],
                                    dates[1],
                                    "1mo")
    X = fdc.MSFT.df.return_log.dropna()
    X = sm.add_constant(X)
    y = fdc.GSPC.df.returns.dropna()    
    model = ms.regression_OLS(X, y)

    # factors
    # 31-Dec-97
    # 31-Dec-02

    # Microsoft
    # S&P500

    # Name	MICROSOFT	S&P 500 				
    # 31-Dec-97	16616.20	1298.82		S&P 500 	MICROSOFT	FITTED
    # 31-Jan-98	19179.30	1313.19		0.011	0.143	0.027
    # 28-Feb-98	21790.70	1407.90		0.070	0.128	0.126
    # 31-Mar-98	23012.00	1480.00		0.050	0.055	0.093
    # 30-Apr-98	23172.70	1494.89		0.010	0.007	0.026
    # 31-May-98	21806.70	1469.19		-0.017	-0.061	-0.021
    # 30-Jun-98	27865.10	1528.87		0.040	0.245	0.076
    # 31-Jul-98	28266.80	1512.59		-0.011	0.014	-0.009
    # 31-Aug-98	24667.20	1293.90		-0.156	-0.136	-0.254
    # 30-Sep-98	28299.00	1376.79		0.062	0.137	0.113
    # 31-Oct-98	27222.30	1488.78		0.078	-0.039	0.140
    # 30-Nov-98	31368.30	1579.01		0.059	0.142	0.108
    # 31-Dec-98	35658.90	1670.01		0.056	0.128	0.103
    # 31-Jan-99	44995.50	1739.84		0.041	0.233	0.078
    # 28-Feb-99	38599.70	1685.77		-0.032	-0.153	-0.045
    # 31-Mar-99	46088.20	1753.21		0.039	0.177	0.075
    # 30-Apr-99	41813.70	1821.11		0.038	-0.097	0.073
    # 31-May-99	41492.30	1778.10		-0.024	-0.008	-0.032
    # 30-Jun-99	46377.50	1876.78		0.054	0.111	0.100
    # 31-Jul-99	44127.70	1818.18		-0.032	-0.050	-0.045
    # 31-Aug-99	47598.80	1809.19		-0.005	0.076	0.000
    # 30-Sep-99	46570.30	1759.59		-0.028	-0.022	-0.038
    # 31-Oct-99	47598.80	1870.94		0.061	0.022	0.112
    # 30-Nov-99	46819.40	1908.97		0.020	-0.017	0.043
    # 31-Dec-99	60036.90	2021.40		0.057	0.249	0.105
    # 31-Jan-00	50330.70	1919.84		-0.052	-0.176	-0.078
    # 29-Feb-00	45959.70	1883.50		-0.019	-0.091	-0.024
    # 31-Mar-00	54637.40	2067.75		0.093	0.173	0.166
    # 30-Apr-00	35867.80	2005.55		-0.031	-0.421	-0.043
    # 31-May-00	32171.80	1964.40		-0.021	-0.109	-0.026
    # 30-Jun-00	41138.70	2012.83		0.024	0.246	0.050
    # 31-Jul-00	35900.00	1981.36		-0.016	-0.136	-0.018
    # 31-Aug-00	35900.00	2104.43		0.060	0.000	0.110
    # 30-Sep-00	31014.80	1993.33		-0.054	-0.146	-0.083
    # 31-Oct-00	35417.90	1984.90		-0.004	0.133	0.002
    # 30-Nov-00	29504.20	1828.42		-0.082	-0.183	-0.130
    # 31-Dec-00	22304.90	1837.36		0.005	-0.280	0.017
    # 31-Jan-01	31400.40	1902.55		0.035	0.342	0.067
    # 28-Feb-01	30339.80	1729.07		-0.096	-0.034	-0.152
    # 31-Mar-01	28122.20	1619.54		-0.065	-0.076	-0.102
    # 30-Apr-01	34839.40	1745.39		0.075	0.214	0.135
    # 31-May-01	35574.70	1757.09		0.007	0.021	0.020
    # 30-Jun-01	37539.10	1714.32		-0.025	0.054	-0.033
    # 31-Jul-01	34037.20	1697.44		-0.010	-0.098	-0.008
    # 31-Aug-01	29337.10	1591.18		-0.065	-0.149	-0.100
    # 30-Sep-01	26313.40	1462.69		-0.084	-0.109	-0.133
    # 31-Oct-01	29902.70	1490.58		0.019	0.128	0.040
    # 30-Nov-01	33019.00	1604.92		0.074	0.099	0.133
    # 31-Dec-01	34068.00	1618.98		0.009	0.031	0.023
    # 31-Jan-02	32761.90	1595.35		-0.015	-0.039	-0.016
    # 28-Feb-02	30000.40	1564.59		-0.019	-0.088	-0.024
    # 31-Mar-02	31013.50	1623.43		0.037	0.033	0.071
    # 30-Apr-02	26873.90	1525.00		-0.063	-0.143	-0.097
    # 31-May-02	26179.70	1513.77		-0.007	-0.026	-0.004
    # 30-Jun-02	28128.60	1405.93		-0.074	0.072	-0.116
    # 31-Jul-02	24673.00	1296.34		-0.081	-0.131	-0.128
    # 31-Aug-02	25238.60	1304.85		0.007	0.023	0.020
    # 30-Sep-02	22492.60	1163.04		-0.115	-0.115	-0.185
    # 31-Oct-02	27496.10	1265.41		0.084	0.201	0.151
    # 30-Nov-02	29661.00	1339.89		0.057	0.076	0.105
    # 31-Dec-02	26585.90	1261.18		-0.061	-0.109	-0.093
