import numpy as np

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
