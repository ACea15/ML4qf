import datetime

index_weblist = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
info_sp500 = ["marketCap", "sector"]

FACTORS = {"famaFrench5Factor": ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
           "momentumFactor": ['MOM']
           }

ASSET_SELECTION_PCT = dict(largest=(0.05, 1),
                           large=(0.25, 2),
                           medium=(0.4, 3),
                           small=(0.25, 2),
                           smallest=(0.05, 1)
                           )
# ASSET_SELECTION_PCT = dict(largest=(0.05, 1),
#                            large=(0.2, 2),
#                            medium=(0.5, 3)
#                            )

# ASSET_SELECTION_NAMES = ['JPM',
#                          'NFLX',
#                          'ANET',
#                          'PH',  
#                          'ROST',
#                          'UDR', 
#                          'RVTY',
#                          'CAG',
#                          'NRG']
# KO  
# LOW 
# GS  
# ALGN
# FICO
# DOV 
# ATO 
# ESS 
# DXC

ASSET_SELECTION_NAMES = None
start_date_factors = "2011-03-31"
end_date_factors =  "2022-12-31"
###
start_date_assets = "2011-02-01"
end_date_assets = "2022-12-01"

start = datetime.datetime.strptime(start_date_assets, "%Y-%m-%d")
end = datetime.datetime.strptime(end_date_assets, "%Y-%m-%d")
days = (end - start).days
train_test_ratio = 0.8
split_data_idx = int(train_test_ratio * days * 12 / 365)

