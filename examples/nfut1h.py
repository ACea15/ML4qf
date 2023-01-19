import pandas as pd
import numpy as np 

# create function to group trade hours
def daypart(hour):
    if hour in [9,10,11]:
        return "morning"
    elif hour in [12,13]:
        return "noon"
    elif hour in [14,15,16,17,18,19]:
        return "afternoon"


# class weight function
def cwts(dfs):
    c0, c1 = np.bincount(dfs)
    w0=(1/c0)*(len(dfs))/2 
    w1=(1/c1)*(len(dfs))/2 
    return {0: w0, 1: w1}

# create function to read locally stored file
def getdata(filename):
    df = pd.read_csv(filename)
    df.datetime = pd.to_datetime(df.datetime)
    df = (
        df.set_index('datetime', drop=True)
        .drop('symbol', axis=1)
    )
    
    # add days
    df['days'] = df.index.day_name()

    # add dayparts
    df['hours'] = df.index.hour
    df['hours'] = df['hours'].apply(daypart)

    return df


df = getdata("/home/ac5015/pCloudDrive/develop/economics/CQF/AdvancedElectives/MachineLearning/MachineLearningWorkshop/data/NFUT1H.csv")

df['ret'] = np.log(df).diff().fillna(0)
df['dir'] = np.where(df.ret.shift(-1)>0,1,0)
df.tail(3)
