import datetime

index_weblist = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
info_sp500 = ["marketCap", "sector"]

FACTORS = {"famaFrench5Factor": ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
           "momentumFactor": ['MOM']
           }

ASSET_SELECTION_PCT = dict(largest=(0.05, 1),
                           large=(0.2, 2),
                           medium=(0.5, 4),
                           small=(0.2, 2),
                           smallest=(0.05, 1)
                           )
# ASSET_SELECTION_PCT = dict(largest=(0.05, 1),
#                            large=(0.2, 2),
#                            medium=(0.5, 3)
#                            )

ASSET_SELECTION_NAMES = ['JPM',     #   AMZN
                         'CVS',#'CMCSA',#'LOW',#'CVS',     #  CVS 
                         'ATVI',    #   ATVI
                         'PH',      #   AFL 
                         'WELL',    #   PAYX
                         'YUM',#'DXCM',    #   WELL
                         'KR',      #  KR  
                         'ATO',     #  ATO 
                         'EQT',#'GEN',     #   GEN 
                         'DXC']     #   RHI 
# KO  
# LOW 
# GS  
# ALGN
# FICO
# DOV 
# ATO 
# ESS 
# DXC

#x JPM  
# CMCSA
# ADI  
# PH   
#x DXCM 
#x TSCO 
#x AEE  
#x CAG  
# CE   
# FRT  

# AMZN
# CVS 
# ATVI
# AFL 
# PAYX
# WELL
# KR  
# ATO 
# GEN 
# RHI 
#ASSET_SELECTION_NAMES = None

start_date_factors = "2000-03-31"
end_date_factors =  "2022-12-31"
###
start_date_assets = "2000-02-01"
end_date_assets = "2022-12-01"

start = datetime.datetime.strptime(start_date_assets, "%Y-%m-%d")
end = datetime.datetime.strptime(end_date_assets, "%Y-%m-%d")
days = (end - start).days
train_test_ratio = 0.9
split_data_idx = int(train_test_ratio * days * 12 / 365)

lambda_mkt = 2.24
lambda_portfolio = [0.1, 2.24, 6]
