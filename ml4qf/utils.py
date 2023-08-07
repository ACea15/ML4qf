import numpy as np
import itertools
import pandas as pd
from collections import namedtuple

# define seed
def set_seeds(libraries, seed=42):
    for li in libraries:
        eval(f"{li}.seed(seed)")

def dict2tuple(x: dict) -> tuple:
    """Converts a dictionary into an equivalent tuple structure

    Parameters
    ----------
    x : dict
        input dictionary

    Returns
    -------
    tuple
        Output tuple

    Examples
    --------
    FIXME: Add docs.

    """

    y = []
    for k, v in x.items():
        z = []
        z.append(k)
        if isinstance(v, dict):
            z.append(dict2tuple(v))
        else:
            z.append(v)
        y.append(tuple(z))
    return tuple(y)

def fix_imbalance(x, *args):
    #print(args)
    #print(args.df['returns'].shift(-1))
    bins = np.where(args[0].df.loc[args[1]]['returns'].shift(-1) > x, 1, 0)
    len_bins = len(bins) 
    len1 = sum(bins)
    len0 =  len_bins - len1
    return ((len1 - len0) / len_bins)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def print_keras_layers(model):
    table = pd.DataFrame(columns=["Name", "Type", "Shape", "Param"])
    for layer in model.layers:
        table = table.append({"Name":layer.name,
                              "Type": layer.__class__.__name__,
                              "Shape": layer.output_shape,
                              "Param": layer.count_params()}, ignore_index=True)
    return table

def create_namedtuple(name, **kwargs):

    NTupled = namedtuple(name, kwargs.keys())
    ntuple = NTupled(**kwargs)
    return ntuple

def trim_df_date(df, start_date=None, end_date=None):
    if start_date is not None:
        start = np.where(df.index == start_date)[0][0]
    else:
        start = None
    if end_date is not None:
        end = np.where(df.index == end_date)[0][0]
    else:
        end = None
    df = df.iloc[start:end]
    return df
